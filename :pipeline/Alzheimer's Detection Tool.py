"""
PupilAD 2.0 — Stage 2 Data Ingestion & Feature Extraction (PLR VARIANT)
Clinical pupillometry pipeline for PLR-only (light reflex) devices
e.g. NeurOptics NPi-200/NPi-300, PLR-3000-style devices

-------------------------------------------------------------------
WHEN TO USE THIS INSTEAD OF ingest_tobii.py
-------------------------------------------------------------------
Use this variant if your pupillometer measures the Pupillary Light
Reflex (PLR) — a brief light flash triggers pupil constriction and
redilation — rather than continuous tracking during cognitive tasks.
This is common with handheld/clinical-grade devices used in neuro-ICU
and ophthalmology settings.

PLR is a well-established, independently validated marker of autonomic
(and by extension LC/brainstem) function, so this isn't a downgrade —
it's a different, clinically standard signal. Some devices output raw
per-sample pupil diameter during the flash (this script expects that);
others only output pre-computed summary metrics (NPi score, latency,
etc.) — if yours does the latter, skip straight to
`build_subject_feature_row_from_summary()` near the bottom.

Because PLR alone doesn't capture cognitive-load-evoked pupillometry,
pair this with a SEPARATE cognitive battery (N-back/Stroop/etc.) scored
independently (accuracy, reaction time) — not pupil-linked — to keep a
cognitive signal in your feature set alongside the autonomic one.

-------------------------------------------------------------------
BEFORE YOU RUN THIS FOR REAL
-------------------------------------------------------------------
Column names and export format are device-specific. Run
inspect_columns() on a real export first and fix COLUMN_MAP below.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass


# =====================================================================
# 1. CONFIG — adjust once you have a real export file
# =====================================================================

COLUMN_MAP = {
    "Timestamp": "timestamp_ms",
    "Pupil Diameter": "pupil_mm",
    "Event": "event",
    "Subject ID": "subject_id",
}

# Marks the light flash onset/offset in the Event column. Many devices
# auto-trigger the flash and log it themselves; check your device's
# export docs for the actual label.
LIGHT_EVENTS = {"flash_on": "LIGHT_ON", "flash_off": "LIGHT_OFF"}

# Number of PLR trials (light flashes) per subject session. Clinical
# protocols often do 2-3 repeated flashes per eye and average results.
TRIALS_PER_SESSION = 3

# Window (ms) after flash onset to search for constriction and recovery
PLR_WINDOW_MS = 4000

# Recovery threshold: fraction of constriction amplitude considered "recovered"
RECOVERY_FRACTION = 0.75  # i.e. T75 recovery time, standard clinical metric

MIN_VALID_PUPIL_MM = 1.5
MAX_VALID_PUPIL_MM = 9.0


# =====================================================================
# 2. LOAD + INSPECT
# =====================================================================

def inspect_columns(filepath: str) -> None:
    df = pd.read_csv(filepath, nrows=20)
    print("Columns found in export:")
    for col in df.columns:
        print(f"  - {col}")
    print("\nFirst few rows:")
    print(df.head())


def load_raw_export(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, low_memory=False)
    missing = [c for c in COLUMN_MAP if c not in df.columns]
    if missing:
        raise ValueError(
            f"Expected columns not found: {missing}\n"
            f"Run inspect_columns() and update COLUMN_MAP."
        )
    df = df.rename(columns=COLUMN_MAP)
    return df[list(COLUMN_MAP.values())]


# =====================================================================
# 3. CLEANING
# =====================================================================

def clean_pupil_signal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bad = (df["pupil_mm"] < MIN_VALID_PUPIL_MM) | (df["pupil_mm"] > MAX_VALID_PUPIL_MM)
    df.loc[bad, "pupil_mm"] = np.nan

    sample_interval_ms = df["timestamp_ms"].diff().median()
    max_gap_samples = int(250 / sample_interval_ms) if sample_interval_ms else 10
    df["pupil_mm"] = df["pupil_mm"].interpolate(
        method="linear", limit=max_gap_samples, limit_area="inside"
    )
    return df


# =====================================================================
# 4. TRIAL SEGMENTATION
# =====================================================================

def segment_plr_trials(df: pd.DataFrame) -> list:
    """
    Split the recording into individual light-flash trials.
    Returns a list of DataFrames, one per trial, each spanning
    [flash_on, flash_on + PLR_WINDOW_MS].
    """
    flash_onsets = df[df["event"] == LIGHT_EVENTS["flash_on"]]["timestamp_ms"].tolist()

    if not flash_onsets:
        print("  [warn] No flash onset markers found")
        return []

    trials = []
    for onset_ts in flash_onsets:
        trial = df[
            (df["timestamp_ms"] >= onset_ts) & (df["timestamp_ms"] <= onset_ts + PLR_WINDOW_MS)
        ].copy()
        trial["t_ms"] = trial["timestamp_ms"] - onset_ts
        trials.append(trial)

    return trials


# =====================================================================
# 5. FEATURE EXTRACTION (standard clinical PLR metrics)
# =====================================================================

@dataclass
class PLRFeatures:
    baseline_diameter_mm: float = np.nan   # pupil size just before flash
    min_diameter_mm: float = np.nan        # max constriction point
    latency_to_onset_ms: float = np.nan    # time from flash to constriction START
    time_to_min_ms: float = np.nan         # time from flash to max constriction
    constriction_amplitude_mm: float = np.nan   # baseline - min (absolute)
    constriction_percent: float = np.nan   # amplitude as % of baseline
    constriction_velocity_mm_s: float = np.nan  # avg speed of constriction
    max_constriction_velocity_mm_s: float = np.nan  # peak speed
    redilation_velocity_mm_s: float = np.nan    # avg speed of recovery
    t75_recovery_ms: float = np.nan        # time to recover 75% of amplitude
    valid_sample_frac: float = np.nan

    def to_dict(self, prefix: str) -> dict:
        return {f"{prefix}_{k}": v for k, v in self.__dict__.items()}


def extract_plr_features(trial: pd.DataFrame) -> PLRFeatures:
    feats = PLRFeatures()
    signal = trial["pupil_mm"]

    if signal.notna().sum() < 5:
        return feats  # insufficient data

    feats.valid_sample_frac = signal.notna().mean()

    # Baseline = mean diameter in the 200ms right before flash (t <= 0)
    pre_flash = trial[trial["t_ms"] <= 0]["pupil_mm"]
    baseline = pre_flash.mean() if not pre_flash.empty else signal.iloc[0]
    feats.baseline_diameter_mm = baseline

    post_flash = trial[trial["t_ms"] > 0]
    if post_flash["pupil_mm"].notna().sum() < 3:
        return feats

    min_idx = post_flash["pupil_mm"].idxmin()
    min_diameter = post_flash.loc[min_idx, "pupil_mm"]
    time_to_min = post_flash.loc[min_idx, "t_ms"]

    feats.min_diameter_mm = min_diameter
    feats.time_to_min_ms = time_to_min
    feats.constriction_amplitude_mm = baseline - min_diameter
    feats.constriction_percent = (
        (feats.constriction_amplitude_mm / baseline * 100) if baseline else np.nan
    )

    # Constriction onset latency: first point where diameter drops
    # meaningfully (>5% of eventual amplitude) below baseline
    threshold = baseline - (0.05 * feats.constriction_amplitude_mm)
    onset_candidates = post_flash[post_flash["pupil_mm"] <= threshold]
    if not onset_candidates.empty:
        feats.latency_to_onset_ms = onset_candidates["t_ms"].iloc[0]

    # Average constriction velocity = amplitude / time to reach min
    if time_to_min and time_to_min > 0:
        feats.constriction_velocity_mm_s = feats.constriction_amplitude_mm / (time_to_min / 1000)

    # Peak instantaneous constriction velocity (max negative slope)
    constrict_phase = post_flash[post_flash["t_ms"] <= time_to_min]
    if len(constrict_phase) > 1:
        d_pupil = constrict_phase["pupil_mm"].diff()
        d_t = constrict_phase["t_ms"].diff() / 1000
        velocities = (d_pupil / d_t).dropna()
        if not velocities.empty:
            feats.max_constriction_velocity_mm_s = velocities.min()  # most negative = fastest constriction

    # Redilation: recovery phase after min diameter
    recovery_phase = post_flash[post_flash["t_ms"] > time_to_min]
    if len(recovery_phase) > 1:
        target_diameter = min_diameter + (RECOVERY_FRACTION * feats.constriction_amplitude_mm)
        recovered = recovery_phase[recovery_phase["pupil_mm"] >= target_diameter]
        if not recovered.empty:
            t75 = recovered["t_ms"].iloc[0] - time_to_min
            feats.t75_recovery_ms = t75
            if t75 > 0:
                feats.redilation_velocity_mm_s = (
                    (target_diameter - min_diameter) / (t75 / 1000)
                )

    return feats


def build_subject_feature_row(subject_id: str, trials: list) -> dict:
    """Average PLR features across repeated trials for one subject."""
    trial_feats = [extract_plr_features(t) for t in trials]
    valid_feats = [f for f in trial_feats if not np.isnan(f.baseline_diameter_mm)]

    row = {"subject_id": subject_id, "n_valid_trials": len(valid_feats)}
    if not valid_feats:
        return row

    # Average each metric across valid trials
    for key in valid_feats[0].__dict__:
        values = [getattr(f, key) for f in valid_feats if not np.isnan(getattr(f, key))]
        row[f"plr_{key}"] = np.mean(values) if values else np.nan

    return row


# =====================================================================
# 5b. ALTERNATE PATH — device outputs pre-computed summary metrics only
# =====================================================================

def build_subject_feature_row_from_summary(filepath: str) -> dict:
    """
    Some clinical PLR devices (e.g. NPi-200) don't export raw traces —
    only a summary row per measurement with fields like NPi score,
    latency, CV (constriction velocity), MCV (max constriction velocity),
    %CH (percent change). Adjust the column names below once you see
    a real export.
    """
    df = pd.read_csv(filepath)
    summary_map = {
        "NPi": "npi_score",
        "Latency": "plr_latency_ms",
        "CV": "plr_constriction_velocity",
        "MCV": "plr_max_constriction_velocity",
        "%CH": "plr_percent_change",
    }
    missing = [c for c in summary_map if c not in df.columns]
    if missing:
        raise ValueError(f"Expected summary columns not found: {missing}")

    row = {"subject_id": Path(filepath).stem}
    for raw_col, std_name in summary_map.items():
        row[std_name] = df[raw_col].mean()  # average across repeated measurements
    return row


# =====================================================================
# 6. PIPELINE ENTRY POINT
# =====================================================================

def process_recording(filepath: str) -> dict:
    print(f"Processing {filepath}")
    raw = load_raw_export(filepath)
    subject_id = raw["subject_id"].iloc[0] if "subject_id" in raw.columns else Path(filepath).stem

    cleaned = clean_pupil_signal(raw)
    trials = segment_plr_trials(cleaned)
    row = build_subject_feature_row(subject_id, trials)
    return row


def process_batch(filepaths: list, output_csv: str = "pupilad_plr_features.csv") -> pd.DataFrame:
    rows = [process_recording(fp) for fp in filepaths]
    feature_df = pd.DataFrame(rows)
    feature_df.to_csv(output_csv, index=False)
    print(f"\nWrote {len(feature_df)} subject rows to {output_csv}")
    return feature_df


if __name__ == "__main__":
    # inspect_columns("raw_exports/subject001_plr.csv")   # run this first!
    #
    # feature_table = process_batch(
    #     filepaths=["raw_exports/subject001_plr.csv"],
    #     output_csv="pupilad_plr_features.csv",
    # )
    print("PLR scaffold ready. See __main__ block for usage once you have a real device export.")
