import argparse

import numpy as np
import pandas as pd

from pupilad.ds003838.loaders import (
    load_ds003838_behaviour,
    load_ds003838_participants,
    load_ds003838_quality,
    load_ds003838_trials,
)
from pupilad.paths import DS003838_DERIVED
from pupilad.signal import linear_slope

LOADS = (5, 9, 13)
MIN_MEMORY_TRIALS_KEPT = 54
MIN_CONTROL_TRIALS_KEPT = 27
ENCODING_DIGITS = tuple(range(1, 6))
OVERLOAD_EARLY_DIGITS = tuple(range(1, 8))
OVERLOAD_LATE_DIGITS = tuple(range(8, 14))
TRIALS_PER_TLX_ADMINISTRATION = 54
TLX_ADMINISTRATIONS = (1, 2, 3)
TLX_SCALES = ("cog", "phys", "speed", "perf", "effort", "dissap")
QUADRATIC_WEIGHTS = {5: 1.0, 9: -2.0, 13: 1.0}
DIGIT_COLUMNS = [f"d{digit}" for digit in range(1, 15)]


def mark_inclusion(quality):
    quality = quality.copy()
    quality["included"] = (quality["n_mem_kept"] >= MIN_MEMORY_TRIALS_KEPT) & (
        quality["n_ctrl_kept"] >= MIN_CONTROL_TRIALS_KEPT
    )
    return quality


def kept_trials(subjects):
    trials = load_ds003838_trials(subjects)
    trials = trials[trials["keep"]].copy()
    trials["d_end"] = [row[f"d{int(row['load'])}"] for _, row in trials.iterrows()]
    trials["tlx_administration"] = (trials["trial"] - 1) // TRIALS_PER_TLX_ADMINISTRATION + 1
    return trials


def cell_means(trials):
    return (
        trials.groupby(["subject", "condition", "load"])[DIGIT_COLUMNS + ["d_end"]]
        .mean()
        .reset_index()
    )


def cell_column(cells, condition, load, column="d_end"):
    selected = cells[(cells["condition"] == condition) & (cells["load"] == load)]
    return selected.set_index("subject")[column]


def memory_minus_control(cells, column="d_end"):
    return pd.DataFrame(
        {
            load: cell_column(cells, "memory", load, column)
            - cell_column(cells, "control", load, column)
            for load in LOADS
        }
    )


def sequence_mean(cells, condition, load):
    selected = cells[(cells["condition"] == condition) & (cells["load"] == load)]
    digit_columns = [f"d{digit}" for digit in range(1, load + 1)]
    return selected.set_index("subject")[digit_columns].mean(axis=1)


def sequence_mean_effect(cells):
    return pd.DataFrame(
        {
            load: sequence_mean(cells, "memory", load) - sequence_mean(cells, "control", load)
            for load in LOADS
        }
    )


def slope_across_loads(effect_by_load):
    loads = np.array(LOADS, dtype=float)
    return effect_by_load[list(LOADS)].apply(
        lambda row: linear_slope(loads, row.to_numpy()), axis=1
    )


def quadratic_contrast(effect_by_load):
    return sum(weight * effect_by_load[load] for load, weight in QUADRATIC_WEIGHTS.items())


def digit_slope(cells, condition, load, digits):
    selected = cells[(cells["condition"] == condition) & (cells["load"] == load)]
    columns = [f"d{digit}" for digit in digits]
    return selected.set_index("subject")[columns].apply(
        lambda row: linear_slope(digits, row.to_numpy()), axis=1
    )


def pooled_encoding_slope(trials, condition):
    means = (
        trials[trials["condition"] == condition]
        .groupby("subject")[[f"d{digit}" for digit in ENCODING_DIGITS]]
        .mean()
    )
    return means.apply(lambda row: linear_slope(ENCODING_DIGITS, row.to_numpy()), axis=1)


def full_sequence_slopes(cells):
    return pd.DataFrame(
        {
            load: digit_slope(cells, "memory", load, tuple(range(1, load + 1)))
            - digit_slope(cells, "control", load, tuple(range(1, load + 1)))
            for load in LOADS
        }
    )


def peak_digit_position(cells, load=13):
    selected = cells[(cells["condition"] == "memory") & (cells["load"] == load)]
    columns = [f"d{digit}" for digit in range(1, load + 1)]
    return selected.set_index("subject")[columns].apply(
        lambda row: int(np.nanargmax(row.to_numpy())) + 1, axis=1
    )


def recall_capacity(subjects, load=13):
    behaviour = load_ds003838_behaviour(subjects)
    return (
        behaviour[behaviour["condition"] == load].groupby("participant_id")["partialScore"].mean()
    )


def tlx_scores(participants, subjects):
    scores = {}
    for administration in TLX_ADMINISTRATIONS:
        scale_columns = [f"NASA_{scale}_{administration}" for scale in TLX_SCALES]
        scores[f"tlx_mental_{administration}"] = participants[f"NASA_cog_{administration}"]
        scores[f"tlx_effort_{administration}"] = participants[f"NASA_effort_{administration}"]
        scores[f"tlx_raw_{administration}"] = participants[scale_columns].mean(axis=1)
    table = pd.DataFrame(scores).reindex(subjects)
    for measure in ("mental", "effort", "raw"):
        table[f"tlx_{measure}_mean"] = table[
            [f"tlx_{measure}_{administration}" for administration in TLX_ADMINISTRATIONS]
        ].mean(axis=1)
    return table


def effect_per_administration(trials):
    cells = (
        trials.groupby(["subject", "tlx_administration", "condition", "load"])["d_end"]
        .mean()
        .unstack("condition")
    )
    per_load = (cells["memory"] - cells["control"]).rename("effect").reset_index()
    return per_load.groupby(["subject", "tlx_administration"])["effect"].mean().reset_index()


def subject_feature_table(trials, cells, quality, participants, subjects):
    end_effect = memory_minus_control(cells)
    mean_effect = sequence_mean_effect(cells)
    sequence_slopes = full_sequence_slopes(cells)
    features = pd.DataFrame(index=pd.Index(subjects, name="subject"))
    for load in LOADS:
        features[f"mem_minus_ctrl_end_{load}"] = end_effect[load]
        features[f"mem_minus_ctrl_sequence_mean_{load}"] = mean_effect[load]
        features[f"sequence_slope_mem_minus_ctrl_{load}"] = sequence_slopes[load]
    features["load_slope_end_mm_per_digit"] = slope_across_loads(end_effect)
    features["load_slope_sequence_mean_mm_per_digit"] = slope_across_loads(mean_effect)
    features["load_quadratic_contrast_end"] = quadratic_contrast(end_effect)
    features["encoding_slope_memory"] = pooled_encoding_slope(trials, "memory")
    features["encoding_slope_control"] = pooled_encoding_slope(trials, "control")
    features["slope_digits_1to7_load13"] = digit_slope(cells, "memory", 13, OVERLOAD_EARLY_DIGITS)
    features["slope_digits_8to13_load13"] = digit_slope(cells, "memory", 13, OVERLOAD_LATE_DIGITS)
    features["peak_position_load13"] = peak_digit_position(cells)
    features["capacity_partial_score_13"] = recall_capacity(subjects)
    features = features.join(tlx_scores(participants, subjects))
    features = features.join(
        quality.set_index("subject")[
            ["eye_used", "valid_frac", "n_mem_kept", "n_ctrl_kept", "mean_baseline_mm"]
        ]
    )
    features["age"] = participants.loc[subjects, "age"]
    return features


def build_features():
    quality = mark_inclusion(load_ds003838_quality())
    subjects = quality.loc[quality["included"], "subject"].tolist()
    trials = kept_trials(subjects)
    cells = cell_means(trials)
    features = subject_feature_table(trials, cells, quality, load_ds003838_participants(), subjects)
    DS003838_DERIVED.mkdir(parents=True, exist_ok=True)
    quality.to_csv(DS003838_DERIVED / "quality.csv", index=False)
    trials.to_csv(DS003838_DERIVED / "kept_trials.csv", index=False)
    cells.to_csv(DS003838_DERIVED / "cell_means.csv", index=False)
    effect_per_administration(trials).to_csv(
        DS003838_DERIVED / "effect_per_tlx_administration.csv", index=False
    )
    features.to_csv(DS003838_DERIVED / "subject_features.csv")
    return quality, features


def main(argv=None):
    argparse.ArgumentParser(prog="pupilad features").parse_args(argv)
    quality, features = build_features()
    print(
        f"{int(quality['included'].sum())} of {len(quality)} participants included; "
        f"{features.shape[1]} features written to {DS003838_DERIVED / 'subject_features.csv'}"
    )


if __name__ == "__main__":
    main()
