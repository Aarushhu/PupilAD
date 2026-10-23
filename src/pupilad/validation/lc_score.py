"""
PupilAD — LC Integrity Score Validation Toolkit
=================================================

Purpose
-------
This script does NOT prove the 70/50 tier cutoffs are diagnostically correct
(that requires external labeled data or your own IRB cohort's pTau181 draws —
see Track B/D of the validation action plan). What it DOES do, using nothing
but your existing task-battery CSVs, is establish two cheaper but genuinely
necessary layers of evidence:

  1. MEASUREMENT VALIDITY — are the pupil features you're extracting even
     computed correctly and physiologically plausible?
  2. CONSTRUCT VALIDITY  — does the LC score (and its underlying features)
     move in the direction established literature says it should, relative
     to task accuracy and cognitive load?

Every literature-derived numeric benchmark in LITERATURE_BENCHMARKS below is
tagged with its source and confidence. Where I only have a *directional*
finding from a paper (not a precise magnitude), it's marked accordingly —
don't let the script silently upgrade a directional finding into a fabricated
number. Fill in exact published magnitudes yourself once you've pulled the
original papers, and update the citation string alongside the number.

Usage
-----
    python validate_lc_score.py --pupil_csv path/to/pupil_data.csv \\
                                 --column_map column_map.json \\
                                 --lc_scores path/to/lc_scores.csv

If no --pupil_csv is given, the script runs on a bundled synthetic demo
dataset so you can see the report format immediately.
"""

import argparse
import json
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# 1. COLUMN MAPPING — edit this (or pass --column_map) to match your actual
#    ingest_tobii.py / ingest_plr.py CSV output schema.
# ---------------------------------------------------------------------------

DEFAULT_COLUMN_MAP = {
    "subject_id": "subject_id",
    "trial_id": "trial_id",
    "task_name": "task_name",          # e.g. n_back, digit_span, stroop, fluency
    "load_level": "load_level",        # cognitive load / difficulty, if applicable
    "timestamp_s": "timestamp_s",      # time within trial, seconds, 0 = stimulus onset
    "pupil_diameter_mm": "pupil_diameter_mm",
    "correct": "correct",              # 1 = correct, 0 = incorrect
    "baseline_window_s": (-1.0, 0.0),  # pre-stimulus window used for baseline
    "response_window_s": (0.0, 4.0),   # post-stimulus window used for task-evoked features
}


# ---------------------------------------------------------------------------
# 2. LITERATURE BENCHMARKS — every entry cites its source. "value" entries are
#    real numbers pulled from published papers; "direction_only" entries mean
#    the paper establishes a direction/relationship, not a specific magnitude
#    you should hold your data to.
# ---------------------------------------------------------------------------

LITERATURE_BENCHMARKS = {
    "plausible_pupil_diameter_mm": {
        "range": (1.5, 9.0),
        "type": "sanity_range",
        "note": (
            "Physiologically plausible human pupil diameter range under typical "
            "lab lighting. Values outside this range almost always indicate a "
            "sensor/tracking artifact, not a real physiological signal — treat "
            "violations as a data-quality bug to fix, not a finding."
        ),
    },
    "constriction_velocity_cutoff_high": {
        "value_mm_s": 4.43,
        "type": "published_cutoff",
        "citation": "Constriction-velocity diagnostic cutoff study (normal-cognition threshold)",
        "reported_at_cutoff": None,
    },
    "constriction_velocity_cutoff_low": {
        "value_mm_s": 3.12,
        "type": "published_cutoff",
        "citation": "Constriction-velocity diagnostic cutoff study (normal-cognition threshold)",
        "reported_at_cutoff": None,
    },
    "pupil_size_change_cutoff": {
        "value_mm": 1.77,
        "type": "published_cutoff",
        "citation": "Pupil-size-change cutoff study",
        "reported_at_cutoff": {
            "sensitivity": 0.613,
            "specificity": 0.80,
            "PPV": 0.4776,
            "NPV": 0.8688,
        },
    },
    "dilation_amplitude_vs_cognitive_load": {
        "type": "direction_only",
        "citation": "Granholm et al. 2017, J Alzheimers Dis 56:1419-1428 (N=918, task-evoked digit span)",
        "expected_direction": (
            "Amnestic single-domain MCI-risk participants show GREATER pupil "
            "dilation than cognitively normal participants at LOWER cognitive "
            "loads, despite normal task accuracy at those loads. "
            "TODO: pull exact effect sizes/mm values from the paper before "
            "using this as a magnitude check rather than a direction check."
        ),
    },
    "dilation_amplitude_tau_association": {
        "type": "direction_only",
        "citation": "Granholm et al. 2021 (tau, not amyloid, associated with pupillary response)",
        "expected_direction": (
            "Pupillary dilation response relates to tau burden specifically, "
            "not amyloid burden. Directly relevant to your pTau181 triage design. "
            "See ADNI_CSF_REFERENCE below for real magnitudes of tau/pTau181 by "
            "diagnosis (not a pupillometry benchmark — no ADNI subject has pupil "
            "data — but useful once you have pTau181 draws on your own subjects)."
        ),
    },
}


# ---------------------------------------------------------------------------
# 2b. ADNI CSF REFERENCE COHORT — real magnitudes computed directly from the
#     local ADNI extract in data/adni/ (UPENNBIOMK_MASTER.csv x DXSUM.csv,
#     inner join on RID+VISCODE, rows with a non-null DIAGNOSIS only).
#     N=3530 visits, computed 2026-09-20.
#
#     IMPORTANT — this is NOT a pupillometry benchmark. No ADNI subject has
#     pupil data, so this cannot validate the LC score directly (see
#     validate_lc_synthetic.py / project notes for why). What it IS: a real
#     normative distribution of CSF ABeta/tau/pTau181 by clinical diagnosis,
#     useful for one thing only — once your own pupillometry subjects have a
#     pTau181 (or tau/ABeta) draw, you can place that value against this
#     distribution to say whether it looks CN-like, MCI-like, or Dementia-like,
#     via check_ptau181_against_adni_reference() below. The correlation that
#     actually validates the LC score (LC score vs. that draw) still has to
#     be computed within your own cohort — ADNI only supplies the yardstick.
# ---------------------------------------------------------------------------

ADNI_CSF_REFERENCE = {
    "source": "Local extract: data/adni/UPENNBIOMK_MASTER.csv joined to DXSUM.csv on (RID, VISCODE)",
    "n_visits": 3530,
    "computed": "2026-09-20",
    "markers": {
        "PTAU": {  # pg/mL, UPenn Luminex xMAP p-Tau181
            "CN":       {"mean": 28.546, "std": 16.412, "median": 23.0, "n": 1055},
            "MCI":      {"mean": 36.529, "std": 20.943, "median": 32.0, "n": 1604},
            "Dementia": {"mean": 43.644, "std": 20.437, "median": 38.6, "n": 847},
            "cohens_d_cn_vs_mci": 0.414,
            "cohens_d_cn_vs_dementia": 0.824,
            "p_cn_vs_mci": 1.11e-30,
            "p_cn_vs_dementia": 9.67e-87,
        },
        "TAU": {  # pg/mL, total tau
            "CN":       {"mean": 73.577, "std": 30.511, "median": 67.60, "n": 1047},
            "MCI":      {"mean": 98.504, "std": 56.057, "median": 83.15, "n": 1606},
            "Dementia": {"mean": 120.725, "std": 61.033, "median": 109.0, "n": 849},
            "cohens_d_cn_vs_mci": 0.523,
            "cohens_d_cn_vs_dementia": 1.009,
            "p_cn_vs_mci": 1.77e-29,
            "p_cn_vs_dementia": 1.64e-91,
        },
        "ABETA": {  # pg/mL, amyloid-beta 1-42 (note: DECREASES with severity)
            "CN":       {"mean": 205.614, "std": 55.522, "median": 215.5, "n": 1058},
            "MCI":      {"mean": 166.924, "std": 51.795, "median": 150.0, "n": 1602},
            "Dementia": {"mean": 139.423, "std": 36.032, "median": 136.0, "n": 852},
            "cohens_d_cn_vs_mci": -0.726,
            "cohens_d_cn_vs_dementia": -1.384,
            "p_cn_vs_mci": 6.96e-65,
            "p_cn_vs_dementia": 1.70e-134,
        },
    },
}


# ---------------------------------------------------------------------------
# 3. FEATURE EXTRACTION — standard pupillometric features from a trial-level
#    pupil-diameter-over-time CSV.
# ---------------------------------------------------------------------------

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


def extract_trial_features(trial_df: pd.DataFrame, cmap: dict) -> TrialFeatures:
    """Compute standard features for a single trial's pupil trace."""
    t = trial_df[cmap["timestamp_s"]].to_numpy()
    d = trial_df[cmap["pupil_diameter_mm"]].to_numpy()

    b_lo, b_hi = cmap["baseline_window_s"]
    r_lo, r_hi = cmap["response_window_s"]

    baseline_mask = (t >= b_lo) & (t <= b_hi)
    response_mask = (t >= r_lo) & (t <= r_hi)

    baseline = float(np.nanmean(d[baseline_mask])) if baseline_mask.any() else np.nan

    if response_mask.any():
        resp_t = t[response_mask]
        resp_d = d[response_mask]
        peak_idx = int(np.nanargmax(resp_d))
        peak_val = float(resp_d[peak_idx])
        peak_latency = float(resp_t[peak_idx] - r_lo)
        amplitude = peak_val - baseline

        # constriction velocity: peak -> end of response window
        if peak_idx < len(resp_d) - 1:
            dt = resp_t[-1] - resp_t[peak_idx]
            dd = peak_val - resp_d[-1]
            velocity = float(dd / dt) if dt > 0 else np.nan
        else:
            velocity = np.nan
    else:
        peak_latency, amplitude, velocity = np.nan, np.nan, np.nan

    row0 = trial_df.iloc[0]
    return TrialFeatures(
        subject_id=row0[cmap["subject_id"]],
        trial_id=row0[cmap["trial_id"]],
        task_name=row0[cmap["task_name"]],
        load_level=row0.get(cmap["load_level"], None),
        correct=int(row0[cmap["correct"]]),
        baseline_diameter_mm=baseline,
        peak_dilation_amplitude_mm=amplitude,
        peak_latency_s=peak_latency,
        mean_constriction_velocity_mm_s=velocity,
    )


def build_feature_table(pupil_df: pd.DataFrame, cmap: dict) -> pd.DataFrame:
    rows = []
    group_cols = [cmap["subject_id"], cmap["trial_id"]]
    for _, trial_df in pupil_df.groupby(group_cols):
        rows.append(extract_trial_features(trial_df, cmap))
    return pd.DataFrame([r.__dict__ for r in rows])


# ---------------------------------------------------------------------------
# 4. VALIDATION CHECKS
# ---------------------------------------------------------------------------

def check_measurement_validity(features: pd.DataFrame) -> list:
    """Layer 1: are the numbers physiologically plausible / free of artifacts?"""
    results = []
    lo, hi = LITERATURE_BENCHMARKS["plausible_pupil_diameter_mm"]["range"]

    # Webcam pixel-diameter data (see task_battery/extract_pupil_from_video.py)
    # runs 20-100+ "mm" once passed through this column — not a data-quality
    # bug, just uncalibrated units. Flag it as skipped/informational instead
    # of a hard FAIL, which would misrepresent a units mismatch as a real
    # measurement problem.
    median_baseline = features["baseline_diameter_mm"].median()
    looks_uncalibrated = pd.notna(median_baseline) and median_baseline > hi * 2

    if looks_uncalibrated:
        results.append({
            "check": "Baseline pupil diameter within plausible physiological range",
            "citation": LITERATURE_BENCHMARKS["plausible_pupil_diameter_mm"]["note"],
            "pass": None,
            "detail": f"SKIPPED — median baseline value ({median_baseline:.1f}) is far outside the "
                      f"{lo}-{hi}mm physiological range, which almost certainly means this data is in "
                      f"uncalibrated units (e.g. webcam pixels), not real millimeters — expected for "
                      f"that data source, not a measurement failure. The Layer 2 checks below remain "
                      f"valid since they're rank-based and don't depend on absolute calibration. Get a "
                      f"real mm-calibrated device before trusting a PASS/FAIL here.",
        })
    else:
        out_of_range = features[
            (features["baseline_diameter_mm"] < lo) | (features["baseline_diameter_mm"] > hi)
        ]
        pct_bad = 100 * len(out_of_range) / max(len(features), 1)
        results.append({
            "check": "Baseline pupil diameter within plausible physiological range",
            "citation": LITERATURE_BENCHMARKS["plausible_pupil_diameter_mm"]["note"],
            "pass": pct_bad < 5.0,
            "detail": f"{pct_bad:.1f}% of trials have baseline diameter outside {lo}-{hi}mm "
                      f"(threshold: fail if >5%).",
        })

    v_lo = LITERATURE_BENCHMARKS["constriction_velocity_cutoff_low"]["value_mm_s"]
    v_hi = LITERATURE_BENCHMARKS["constriction_velocity_cutoff_high"]["value_mm_s"]
    if looks_uncalibrated:
        results.append({
            "check": "Constriction velocity in same order of magnitude as published cutoffs",
            "citation": LITERATURE_BENCHMARKS["constriction_velocity_cutoff_low"]["citation"],
            "pass": None,
            "detail": "SKIPPED — same uncalibrated-units issue as the baseline-diameter check above; "
                      "this cutoff assumes real mm/s.",
        })
    else:
        velocities = features["mean_constriction_velocity_mm_s"].dropna()
        frac_in_published_band = ((velocities >= v_lo * 0.5) & (velocities <= v_hi * 2)).mean() if len(velocities) else 0
        results.append({
            "check": "Constriction velocity in same order of magnitude as published cutoffs",
            "citation": LITERATURE_BENCHMARKS["constriction_velocity_cutoff_low"]["citation"],
            "pass": frac_in_published_band > 0.8,
            "detail": f"{frac_in_published_band*100:.1f}% of trials fall within a generous band "
                      f"around the published {v_lo}-{v_hi} mm/s cutoffs. A low number here usually "
                      f"means a units mismatch (e.g., mm/s vs px/s) rather than a real physiological finding.",
        })
    return results


def check_construct_validity(features: pd.DataFrame) -> list:
    """Layer 2: does the signal move the way the literature predicts?"""
    results = []

    # Accuracy vs. dilation amplitude, split by load if available
    valid = features.dropna(subset=["peak_dilation_amplitude_mm", "correct"])
    if len(valid) >= 10:
        correct_amp = valid.loc[valid["correct"] == 1, "peak_dilation_amplitude_mm"]
        incorrect_amp = valid.loc[valid["correct"] == 0, "peak_dilation_amplitude_mm"]
        if len(correct_amp) > 1 and len(incorrect_amp) > 1:
            t_stat, p_val = stats.ttest_ind(correct_amp, incorrect_amp, equal_var=False)
            results.append({
                "check": "Dilation amplitude differs between correct/incorrect trials",
                "citation": LITERATURE_BENCHMARKS["dilation_amplitude_vs_cognitive_load"]["citation"],
                "pass": p_val < 0.10,  # exploratory threshold given likely small pilot N
                "detail": f"t={t_stat:.2f}, p={p_val:.3f} "
                          f"(mean amp correct={correct_amp.mean():.3f}mm, "
                          f"incorrect={incorrect_amp.mean():.3f}mm). "
                          f"NOTE: direction alone isn't diagnostic — Granholm's finding is that "
                          f"dilation can be elevated in MCI-risk participants EVEN WHEN accuracy "
                          f"is normal, so a null result here doesn't necessarily invalidate the "
                          f"score. Use this as a data-quality signal, not a pass/fail on LC validity.",
            })
    else:
        results.append({
            "check": "Dilation amplitude vs. accuracy",
            "citation": "n/a",
            "pass": None,
            "detail": "Not enough trials with both dilation and accuracy data to test (need >=10).",
        })

    # Monotonic trend with load, if load_level present and numeric
    if "load_level" in features.columns and features["load_level"].notna().any():
        try:
            load_numeric = pd.to_numeric(features["load_level"], errors="coerce")
            sub = features.assign(_load=load_numeric).dropna(subset=["_load", "peak_dilation_amplitude_mm"])
            if len(sub) >= 10:
                rho, p_val = stats.spearmanr(sub["_load"], sub["peak_dilation_amplitude_mm"])
                results.append({
                    "check": "Dilation amplitude increases with cognitive load",
                    "citation": LITERATURE_BENCHMARKS["dilation_amplitude_vs_cognitive_load"]["citation"],
                    "pass": rho > 0 and p_val < 0.10,
                    "detail": f"Spearman rho={rho:.2f}, p={p_val:.3f}. Expected: positive correlation "
                              f"(more dilation at higher cognitive load) per task-evoked pupillometry literature.",
                })
        except Exception as e:
            results.append({
                "check": "Dilation amplitude vs. load",
                "citation": "n/a",
                "pass": None,
                "detail": f"Could not compute — {e}",
            })

    return results


def check_lc_score_correlation(features: pd.DataFrame, lc_scores: pd.DataFrame) -> list:
    """Layer 2 (continued): does the LSTM's LC score itself track accuracy sensibly."""
    results = []
    merged = features.merge(lc_scores, on="subject_id", how="inner")
    if len(merged) < 5:
        results.append({
            "check": "LC score vs. subject-level accuracy",
            "citation": "n/a",
            "pass": None,
            "detail": "Not enough matched subjects between feature table and LC score file.",
        })
        return results

    subj_acc = merged.groupby("subject_id")["correct"].mean()
    subj_score = merged.groupby("subject_id")["lc_score"].first()
    rho, p_val = stats.spearmanr(subj_acc, subj_score)
    results.append({
        "check": "LC score correlates with subject-level task accuracy",
        "citation": "General construct-validity expectation, not a specific paper",
        "pass": None,  # informational — direction depends on task load design
        "detail": f"Spearman rho={rho:.2f}, p={p_val:.3f} across {len(subj_acc)} subjects. "
                  f"Interpret carefully: a high positive correlation with accuracy on easy trials "
                  f"is expected; on high-load trials, LC-risk subjects may show HIGH dilation "
                  f"(low LC score) despite still-normal accuracy (per Granholm). Don't treat "
                  f"'low correlation with accuracy' alone as a failure.",
    })
    return results


def check_ptau181_against_adni_reference(ptau181_value: float) -> list:
    """
    Layer 3: place a real pTau181 draw from one of YOUR subjects against the
    ADNI CSF reference cohort (see ADNI_CSF_REFERENCE) to say whether it looks
    CN-like, MCI-like, or Dementia-like. This does NOT validate the LC score
    by itself — it only interprets the biomarker. The actual validating step
    is correlating LC score against this same subject's draw across your
    cohort, once you have enough subjects with both.
    """
    ref = ADNI_CSF_REFERENCE["markers"]["PTAU"]
    results = []

    z_scores = {}
    for group in ("CN", "MCI", "Dementia"):
        mean, std = ref[group]["mean"], ref[group]["std"]
        z_scores[group] = abs(ptau181_value - mean) / std

    closest_group = min(z_scores, key=z_scores.get)

    detail_lines = [
        f"{group}: ADNI mean={ref[group]['mean']:.1f} pg/mL (sd={ref[group]['std']:.1f}, "
        f"n={ref[group]['n']}), this value is {z_scores[group]:.2f} SD away"
        for group in ("CN", "MCI", "Dementia")
    ]
    results.append({
        "check": f"pTau181={ptau181_value:.1f} pg/mL vs. ADNI CN/MCI/Dementia reference",
        "citation": ADNI_CSF_REFERENCE["source"] + f" (n={ADNI_CSF_REFERENCE['n_visits']} visits, "
                    f"computed {ADNI_CSF_REFERENCE['computed']})",
        "pass": None,  # informational only — proximity to a group isn't a diagnosis
        "detail": (
            f"Closest to the {closest_group} reference group. " + "; ".join(detail_lines) +
            ". This is a normative comparison only, not a diagnosis, and ADNI's CSF assay "
            "may not be directly comparable to a different lab's assay/units — check before "
            "treating these numbers as calibrated to your own draws."
        ),
    })
    return results


# ---------------------------------------------------------------------------
# 5. REPORT
# ---------------------------------------------------------------------------

def print_report(all_checks: list):
    print("=" * 78)
    print("PupilAD LC Score — Validation Report")
    print("=" * 78)
    for section, checks in all_checks:
        print(f"\n--- {section} ---")
        for c in checks:
            status = {"True": "PASS", "False": "FAIL", "None": "INFO"}[str(c["pass"])]
            print(f"[{status}] {c['check']}")
            print(f"   Source: {c['citation']}")
            print(f"   {c['detail']}\n")


# ---------------------------------------------------------------------------
# 6. SYNTHETIC DEMO DATA (used only if no --pupil_csv is supplied)
# ---------------------------------------------------------------------------

def make_demo_data(n_subjects=20, n_trials=8, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subjects):
        subj_id = f"S{s:03d}"
        risk = rng.random() < 0.3  # 30% "at-risk" synthetic profile
        for tr in range(n_trials):
            load = rng.integers(1, 4)
            baseline = rng.normal(4.5, 0.4)
            true_amp = 0.15 * load + (0.25 if risk and load <= 2 else 0.0) + rng.normal(0, 0.05)
            correct = rng.random() > (0.05 * load - (0.1 if risk else 0))
            t = np.linspace(-1, 4, 50)
            trace = baseline + np.where(
                t < 0, 0,
                true_amp * (1 - np.exp(-t / 0.8)) * np.exp(-0.05 * t)
            ) + rng.normal(0, 0.02, size=t.shape)
            for ti, di in zip(t, trace):
                rows.append({
                    "subject_id": subj_id, "trial_id": f"{subj_id}_T{tr}",
                    "task_name": "n_back", "load_level": load,
                    "timestamp_s": ti, "pupil_diameter_mm": di,
                    "correct": int(correct),
                })
    df = pd.DataFrame(rows)
    lc_scores = pd.DataFrame({
        "subject_id": [f"S{s:03d}" for s in range(n_subjects)],
        "lc_score": rng.normal(65, 15, size=n_subjects).clip(0, 100),
    })
    return df, lc_scores


# ---------------------------------------------------------------------------
# 7. MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Validate PupilAD LC score against literature + internal consistency.")
    parser.add_argument("--pupil_csv", type=str, default=None,
                         help="Path to trial-level pupil CSV from ingest_tobii.py/ingest_plr.py. "
                              "If omitted, runs on synthetic demo data.")
    parser.add_argument("--lc_scores", type=str, default=None,
                         help="Path to CSV with columns subject_id, lc_score.")
    parser.add_argument("--column_map", type=str, default=None,
                         help="Path to JSON file overriding DEFAULT_COLUMN_MAP.")
    parser.add_argument("--ptau181", type=float, default=None,
                         help="A real pTau181 draw (pg/mL) from one of your subjects. If given, "
                              "reports where it falls relative to the ADNI CN/MCI/Dementia "
                              "reference cohort (see ADNI_CSF_REFERENCE) — informational only, "
                              "not a validation of the LC score by itself.")
    args = parser.parse_args()

    cmap = dict(DEFAULT_COLUMN_MAP)
    if args.column_map:
        with open(args.column_map) as f:
            cmap.update(json.load(f))

    if args.pupil_csv:
        pupil_df = pd.read_csv(args.pupil_csv)
        lc_scores = pd.read_csv(args.lc_scores) if args.lc_scores else None
    else:
        print("No --pupil_csv given — running on synthetic demo data so you can see the report format.\n")
        pupil_df, lc_scores = make_demo_data()

    features = build_feature_table(pupil_df, cmap)

    all_checks = []
    all_checks.append(("Layer 1: Measurement validity", check_measurement_validity(features)))
    all_checks.append(("Layer 2: Construct validity (internal)", check_construct_validity(features)))
    if lc_scores is not None:
        all_checks.append(("Layer 2: Construct validity (LC score)", check_lc_score_correlation(features, lc_scores)))
    if args.ptau181 is not None:
        all_checks.append(("Layer 3: External biomarker context (ADNI reference)",
                            check_ptau181_against_adni_reference(args.ptau181)))

    print_report(all_checks)

    features.to_csv("extracted_features.csv", index=False)
    print("Extracted per-trial features written to extracted_features.csv")


if __name__ == "__main__":
    main()
