"""
PupilAD — Synthetic Pipeline Validation
========================================

Purpose
-------
Before spending a real pilot session's time on validate_lc_score.py (which
needs real, time-aligned pupil + accuracy data we don't have yet — see
project notes), this script answers a narrower, cheaper question first:

    Can the REAL production pipeline (Synthetic Tobii Generator.py ->
    Ingest Tobii Script.py -> Alzheimer's Detection LSTM.py) recover a
    KNOWN, designed-in signal at all, out of sample, with what confidence,
    and at what cohort size does that confidence stop improving?

This is a pipeline/architecture sanity check, NOT a validity claim. The
synthetic generator's "healthy" vs "mci_like" profiles are stylized curve
shapes chosen by the generator's author, not derived from patient data —
recovering them proves the feature extraction + LSTM can learn a real
signal from a raw trace and generalize to held-out subjects. It does NOT
prove the 70/50 tier cutoffs, or the LC score itself, are diagnostically
meaningful. That still requires real labeled data (see validate_lc_score.py
docstring, Track B/D of the validation action plan).

Method
------
For each cohort size:
  1. Generate a fresh synthetic cohort (Synthetic Tobii Generator.py).
  2. Run it through the REAL ingestion pipeline (Ingest Tobii Script.py) —
     same feature extraction the real pipeline uses on Tobii exports.
  3. Split subjects into train/held-out BEFORE fitting the scaler or model
     (no leakage).
  4. Train the REAL LSTM architecture (Alzheimer's Detection LSTM.py) on
     the train split, using the CLEAN ground-truth profile label (not the
     noisy proxy label train_on_synthetic() uses for its demo — that noise
     was for illustrating output format, not for a real recovery test).
  5. On the held-out split only, compute:
       - AUC (does the score rank healthy above mci_like subjects?)
       - Mann-Whitney U p-value (is the separation itself significant?)
       - Accuracy at the score>=50 cutoff (matches the LSTM's own "watch"
         boundary)
  6. Repeat several times per cohort size (fresh synthetic draw + fresh
     train/test split each time) to get a distribution, not one lucky/
     unlucky split.

Then compare cohort sizes to each other (independent-samples test on the
held-out AUCs) to find the point where adding more synthetic subjects stops
producing a statistically significant improvement — i.e. where the
architecture's recoverable signal has plateaued for this feature set.

Usage
-----
    .venv/bin/python validate_lc_synthetic.py \\
        --cohort-sizes 20 40 80 --repeats 4 --epochs 30
"""

import argparse
import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent
PIPELINE_DIR = REPO_ROOT / ":pipeline"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


synth_gen = _load_module(PIPELINE_DIR / "Synthetic Tobii Generator.py", "pupilad_synth_gen")
ingest = _load_module(PIPELINE_DIR / "Ingest Tobii Script.py", "pupilad_ingest_tobii")
lstm_mod = _load_module(PIPELINE_DIR / "Alzheimer's Detection LSTM.py", "pupilad_lstm")


# ---------------------------------------------------------------------------
# Cohort generation + real ingestion
# ---------------------------------------------------------------------------

def build_cohort_features(n_healthy: int, n_mci: int, seed: int, workdir: Path,
                           difficulty: float = 0.0) -> pd.DataFrame:
    """Generate one synthetic cohort and run it through the real ingest pipeline."""
    export_dir = workdir / f"synthetic_exports_seed{seed}"
    export_dir.mkdir(parents=True, exist_ok=True)
    filepaths = []

    for i in range(n_healthy):
        subject_id = f"SYN_HEALTHY_{seed}_{i:03d}"
        df = synth_gen.generate_subject_session(subject_id, profile="healthy", seed=seed * 100_000 + i,
                                                  difficulty=difficulty)
        fp = export_dir / f"{subject_id}.tsv"
        df.to_csv(fp, sep="\t", index=False)
        filepaths.append(str(fp))

    for i in range(n_mci):
        subject_id = f"SYN_MCILIKE_{seed}_{i:03d}"
        df = synth_gen.generate_subject_session(subject_id, profile="mci_like", seed=seed * 100_000 + 50_000 + i,
                                                  difficulty=difficulty)
        fp = export_dir / f"{subject_id}.tsv"
        df.to_csv(fp, sep="\t", index=False)
        filepaths.append(str(fp))

    feature_rows = [ingest.process_recording(fp) for fp in filepaths]
    return pd.DataFrame(feature_rows)


# ---------------------------------------------------------------------------
# Train on a split, evaluate recovery on the held-out split
# ---------------------------------------------------------------------------

def evaluate_cohort(feature_df: pd.DataFrame, seed: int, epochs: int) -> dict:
    y_true = feature_df["subject_id"].str.contains("HEALTHY").astype(int).to_numpy()
    X = lstm_mod.reshape_features_to_sequence(feature_df)

    idx = np.arange(len(feature_df))
    idx_train, idx_test = train_test_split(
        idx, test_size=0.3, random_state=seed, stratify=y_true
    )

    n_subj, n_tasks, n_feats = X.shape
    scaler = StandardScaler()
    scaler.fit(X[idx_train].reshape(-1, n_feats))
    X_scaled = scaler.transform(X.reshape(-1, n_feats)).reshape(n_subj, n_tasks, n_feats)

    model = lstm_mod.build_lstm_model(n_tasks, n_feats)
    model.fit(
        X_scaled[idx_train], y_true[idx_train].astype(float),
        epochs=epochs, batch_size=8, verbose=0,
    )

    pred_test = model.predict(X_scaled[idx_test], verbose=0).flatten()
    y_test = y_true[idx_test]

    healthy_scores = pred_test[y_test == 1]
    mci_scores = pred_test[y_test == 0]

    if len(set(y_test)) > 1:
        auc = roc_auc_score(y_test, pred_test)
        u_stat, p_val = stats.mannwhitneyu(healthy_scores, mci_scores, alternative="greater")
    else:
        auc, p_val = np.nan, np.nan

    pred_label = (pred_test >= 0.5).astype(int)
    accuracy_at_50 = float((pred_label == y_test).mean())

    return {
        "seed": seed,
        "n_train": len(idx_train),
        "n_test": len(idx_test),
        "held_out_auc": auc,
        "mannwhitney_p": p_val,
        "held_out_accuracy_at_50": accuracy_at_50,
        "mean_score_true_healthy": float(healthy_scores.mean() * 100) if len(healthy_scores) else np.nan,
        "mean_score_true_mci": float(mci_scores.mean() * 100) if len(mci_scores) else np.nan,
    }


# ---------------------------------------------------------------------------
# Scaling curve across cohort sizes
# ---------------------------------------------------------------------------

def run_scaling_curve(cohort_sizes, n_repeats: int, epochs: int, workdir: Path,
                       difficulty: float = 0.0) -> pd.DataFrame:
    records = []
    for size in cohort_sizes:
        n_healthy = size // 2
        n_mci = size - n_healthy
        for rep in range(n_repeats):
            seed = size * 1000 + rep
            features = build_cohort_features(n_healthy, n_mci, seed, workdir, difficulty)
            result = evaluate_cohort(features, seed, epochs)
            result["cohort_size"] = size
            result["difficulty"] = difficulty
            records.append(result)
            print(
                f"[size={size:>4} rep={rep}] "
                f"held-out AUC={result['held_out_auc']:.3f}  "
                f"p={result['mannwhitney_p']:.4f}  "
                f"acc@50={result['held_out_accuracy_at_50']:.2f}  "
                f"(healthy->{result['mean_score_true_healthy']:.1f}, "
                f"mci_like->{result['mean_score_true_mci']:.1f})"
            )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Difficulty sweep — find where the profiles stop being trivially separable
# ---------------------------------------------------------------------------

def run_difficulty_sweep(difficulties, cohort_size: int, n_repeats: int, epochs: int,
                          workdir: Path) -> pd.DataFrame:
    records = []
    n_healthy = cohort_size // 2
    n_mci = cohort_size - n_healthy
    for difficulty in difficulties:
        for rep in range(n_repeats):
            seed = int(difficulty * 10_000) * 1000 + rep
            features = build_cohort_features(n_healthy, n_mci, seed, workdir, difficulty)
            result = evaluate_cohort(features, seed, epochs)
            result["difficulty"] = difficulty
            result["cohort_size"] = cohort_size
            records.append(result)
            print(
                f"[difficulty={difficulty:.2f} rep={rep}] "
                f"held-out AUC={result['held_out_auc']:.3f}  "
                f"p={result['mannwhitney_p']:.4f}  "
                f"acc@50={result['held_out_accuracy_at_50']:.2f}  "
                f"(healthy->{result['mean_score_true_healthy']:.1f}, "
                f"mci_like->{result['mean_score_true_mci']:.1f})"
            )
    return pd.DataFrame(records)


def print_difficulty_report(results: pd.DataFrame):
    print("\n" + "=" * 78)
    print("Difficulty sweep — where does the pipeline stop separating cleanly?")
    print("=" * 78)
    summary = results.groupby("difficulty")["held_out_auc"].agg(["mean", "std", "count"])
    print(summary.to_string())
    below_95 = summary[summary["mean"] < 0.95]
    if len(below_95):
        print(f"\nAUC first drops below 0.95 at difficulty={below_95.index[0]}.")
    else:
        print("\nAUC never dropped below 0.95 across the sweep — try higher difficulty values.")


def print_plateau_report(results: pd.DataFrame):
    print("\n" + "=" * 78)
    print("Scaling summary — does more synthetic data still help?")
    print("=" * 78)

    sizes = sorted(results["cohort_size"].unique())
    summary = results.groupby("cohort_size")["held_out_auc"].agg(["mean", "std", "count"])
    print(summary.to_string())

    print("\nPairwise comparison of consecutive cohort sizes (held-out AUC):")
    for a, b in zip(sizes[:-1], sizes[1:]):
        auc_a = results.loc[results["cohort_size"] == a, "held_out_auc"].dropna()
        auc_b = results.loc[results["cohort_size"] == b, "held_out_auc"].dropna()
        if len(auc_a) > 1 and len(auc_b) > 1:
            t_stat, p_val = stats.ttest_ind(auc_b, auc_a, equal_var=False)
            direction = "improved" if auc_b.mean() > auc_a.mean() else "did not improve"
            sig = "SIGNIFICANT" if p_val < 0.05 else "not significant"
            print(
                f"  {a:>4} -> {b:>4} subjects: AUC {auc_a.mean():.3f} -> {auc_b.mean():.3f} "
                f"({direction}), t={t_stat:.2f}, p={p_val:.3f} [{sig}]"
            )
        else:
            print(f"  {a:>4} -> {b:>4} subjects: not enough repeats to compare")

    print(
        "\nInterpretation guide: once a step stops being SIGNIFICANT, further "
        "synthetic scale isn't the bottleneck anymore — the ceiling is set by "
        "the feature set / architecture, not sample size. That's the point to "
        "stop tuning against synthetic data and move to real labeled data "
        "instead of chasing a higher number here."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate the real ingest->LSTM pipeline's ability to recover "
                     "a known synthetic ground-truth signal, out of sample."
    )
    parser.add_argument("--mode", choices=["scaling", "difficulty"], default="scaling",
                         help="'scaling': cohort-size plateau curve at a fixed difficulty. "
                              "'difficulty': sweep difficulty at a fixed cohort size to find "
                              "where AUC stops being trivially 1.0.")
    parser.add_argument("--cohort-sizes", type=int, nargs="+", default=[20, 40, 80])
    parser.add_argument("--cohort-size", type=int, default=40,
                         help="Fixed cohort size used in --mode difficulty.")
    parser.add_argument("--difficulty", type=float, default=0.0,
                         help="Fixed difficulty (0.0-1.0) used in --mode scaling.")
    parser.add_argument("--difficulties", type=float, nargs="+",
                         default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--output", type=str, default="synthetic_validation_results.csv")
    parser.add_argument("--keep-workdir", action="store_true",
                         help="Don't delete the generated synthetic .tsv exports afterward.")
    args = parser.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="pupilad_synth_"))
    print(f"Generating synthetic cohorts under {workdir}\n")

    try:
        if args.mode == "scaling":
            results = run_scaling_curve(args.cohort_sizes, args.repeats, args.epochs, workdir, args.difficulty)
        else:
            results = run_difficulty_sweep(args.difficulties, args.cohort_size, args.repeats, args.epochs, workdir)
    finally:
        if not args.keep_workdir:
            shutil.rmtree(workdir, ignore_errors=True)

    results.to_csv(args.output, index=False)
    print(f"\nPer-run results written to {args.output}")

    if args.mode == "scaling":
        print_plateau_report(results)
    else:
        print_difficulty_report(results)


if __name__ == "__main__":
    main()
