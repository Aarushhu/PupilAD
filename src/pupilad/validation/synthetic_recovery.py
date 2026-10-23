import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pupilad.model.lstm import build_lstm_model, features_to_sequence
from pupilad.paths import RESULTS_DIR
from pupilad.tobii.ingest import process_batch
from pupilad.tobii.synthetic import generate_subject_session

SYNTHETIC_RESULTS = RESULTS_DIR / "synthetic"
TEST_FRACTION = 0.3
BATCH_SIZE = 8
SCORE_CUTOFF = 0.5
AUC_PLATEAU_LEVEL = 0.95


def write_cohort(n_healthy, n_mci, seed, workdir, difficulty):
    export_dir = workdir / f"synthetic_exports_seed{seed}"
    export_dir.mkdir(parents=True, exist_ok=True)
    cohort = [
        (f"SYN_HEALTHY_{seed}_{i:03d}", "healthy", seed * 100_000 + i) for i in range(n_healthy)
    ]
    cohort += [
        (f"SYN_MCILIKE_{seed}_{i:03d}", "mci_like", seed * 100_000 + 50_000 + i)
        for i in range(n_mci)
    ]
    paths = []
    for subject_id, profile, subject_seed in cohort:
        path = export_dir / f"{subject_id}.tsv"
        generate_subject_session(subject_id, profile, subject_seed, difficulty).to_csv(
            path, sep="\t", index=False
        )
        paths.append(path)
    return paths


def scaled_sequences(sequences, train_index):
    n_subjects, n_tasks, n_features = sequences.shape
    scaler = StandardScaler().fit(sequences[train_index].reshape(-1, n_features))
    return scaler.transform(sequences.reshape(-1, n_features)).reshape(
        n_subjects, n_tasks, n_features
    )


def held_out_metrics(predictions, labels):
    healthy = predictions[labels == 1]
    mci_like = predictions[labels == 0]
    if len(set(labels)) > 1:
        auc = roc_auc_score(labels, predictions)
        p_value = stats.mannwhitneyu(healthy, mci_like, alternative="greater").pvalue
    else:
        auc, p_value = np.nan, np.nan
    return {
        "held_out_auc": auc,
        "mannwhitney_p": p_value,
        "held_out_accuracy_at_50": float(
            ((predictions >= SCORE_CUTOFF).astype(int) == labels).mean()
        ),
        "mean_score_true_healthy": float(healthy.mean() * 100) if len(healthy) else np.nan,
        "mean_score_true_mci": float(mci_like.mean() * 100) if len(mci_like) else np.nan,
    }


def evaluate_cohort(feature_table, seed, epochs):
    labels = feature_table["subject_id"].str.contains("HEALTHY").astype(int).to_numpy()
    sequences = features_to_sequence(feature_table)
    train_index, test_index = train_test_split(
        np.arange(len(labels)), test_size=TEST_FRACTION, random_state=seed, stratify=labels
    )
    sequences = scaled_sequences(sequences, train_index)
    model = build_lstm_model(sequences.shape[1], sequences.shape[2])
    model.fit(
        sequences[train_index],
        labels[train_index].astype(float),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        verbose=0,
    )
    predictions = model.predict(sequences[test_index], verbose=0).flatten()
    return {
        "seed": seed,
        "n_train": len(train_index),
        "n_test": len(test_index),
        **held_out_metrics(predictions, labels[test_index]),
    }


def run_condition(cohort_size, difficulty, seed, epochs, workdir):
    n_healthy = cohort_size // 2
    paths = write_cohort(n_healthy, cohort_size - n_healthy, seed, workdir, difficulty)
    result = evaluate_cohort(process_batch(paths), seed, epochs)
    result.update({"cohort_size": cohort_size, "difficulty": difficulty})
    print(
        f"size={cohort_size} difficulty={difficulty:.2f} seed={seed} "
        f"AUC={result['held_out_auc']:.3f} p={result['mannwhitney_p']:.4f}",
        flush=True,
    )
    return result


def scaling_curve(cohort_sizes, repeats, epochs, difficulty, workdir):
    return pd.DataFrame(
        [
            run_condition(size, difficulty, size * 1000 + repeat, epochs, workdir)
            for size in cohort_sizes
            for repeat in range(repeats)
        ]
    )


def difficulty_sweep(difficulties, cohort_size, repeats, epochs, workdir):
    return pd.DataFrame(
        [
            run_condition(
                cohort_size, difficulty, int(difficulty * 10_000) * 1000 + repeat, epochs, workdir
            )
            for difficulty in difficulties
            for repeat in range(repeats)
        ]
    )


def summarize(results, group_column):
    summary = results.groupby(group_column)["held_out_auc"].agg(["mean", "std", "count"])
    print(summary.to_string())
    return summary


def compare_consecutive_sizes(results):
    sizes = sorted(results["cohort_size"].unique())
    for smaller, larger in zip(sizes[:-1], sizes[1:]):
        auc_small = results.loc[results["cohort_size"] == smaller, "held_out_auc"].dropna()
        auc_large = results.loc[results["cohort_size"] == larger, "held_out_auc"].dropna()
        if len(auc_small) > 1 and len(auc_large) > 1:
            test = stats.ttest_ind(auc_large, auc_small, equal_var=False)
            print(
                f"{smaller} -> {larger}: AUC {auc_small.mean():.3f} -> {auc_large.mean():.3f}, "
                f"t={test.statistic:.2f}, p={test.pvalue:.3f}"
            )


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pupilad train")
    parser.add_argument("--mode", choices=["scaling", "difficulty"], default="scaling")
    parser.add_argument("--cohort-sizes", type=int, nargs="+", default=[20, 40, 80])
    parser.add_argument("--cohort-size", type=int, default=40)
    parser.add_argument("--difficulty", type=float, default=0.0)
    parser.add_argument(
        "--difficulties", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    workdir = Path(tempfile.mkdtemp(prefix="pupilad_synth_"))
    try:
        if args.mode == "scaling":
            results = scaling_curve(
                args.cohort_sizes, args.repeats, args.epochs, args.difficulty, workdir
            )
        else:
            results = difficulty_sweep(
                args.difficulties, args.cohort_size, args.repeats, args.epochs, workdir
            )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    SYNTHETIC_RESULTS.mkdir(parents=True, exist_ok=True)
    output = args.output or SYNTHETIC_RESULTS / f"recovery_{args.mode}.csv"
    results.to_csv(output, index=False)
    if args.mode == "scaling":
        summarize(results, "cohort_size")
        compare_consecutive_sizes(results)
    else:
        summary = summarize(results, "difficulty")
        below = summary[summary["mean"] < AUC_PLATEAU_LEVEL]
        if len(below):
            print(f"Mean AUC first drops below {AUC_PLATEAU_LEVEL} at difficulty {below.index[0]}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
