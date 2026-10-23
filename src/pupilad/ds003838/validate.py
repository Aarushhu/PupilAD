import argparse
import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from pupilad.ds003838.features import LOADS
from pupilad.ds003838.figures import (
    plot_load_dilation,
    plot_load_slope_and_tlx,
    plot_reliability,
)
from pupilad.ds003838.loaders import load_ds003838_participants, load_ds003838_traces
from pupilad.paths import DS003838_DERIVED, DS003838_RESULTS
from pupilad.signal import linear_slope
from pupilad.stats import (
    bootstrap_ci,
    friedman_effect,
    holm_adjust,
    paired_effect,
    repeated_measures_correlation,
    spearman_effect,
)

LOAD_PAIRS = ((5, 9), (9, 13), (5, 13))
TLX_MEASURES = ("mental", "effort", "raw")
ENCODING_COLUMNS = [f"d{digit}" for digit in range(1, 6)]


def load_derived():
    return (
        pd.read_csv(DS003838_DERIVED / "quality.csv"),
        pd.read_csv(DS003838_DERIVED / "kept_trials.csv"),
        pd.read_csv(DS003838_DERIVED / "subject_features.csv").set_index("subject"),
        pd.read_csv(DS003838_DERIVED / "effect_per_tlx_administration.csv"),
    )


def with_holm(results):
    for key, adjusted in zip(results, holm_adjust([results[key]["p"] for key in results])):
        results[key]["p_holm"] = float(adjusted)
    return results


def sample_description(quality, trials, features):
    participants = load_ds003838_participants()
    included = features.index.tolist()
    return {
        "n_with_pupil_data": int(len(quality)),
        "n_included": len(included),
        "excluded": quality.loc[~quality["included"], "subject"].tolist(),
        "trials_kept": int(len(trials)),
        "age_range": [int(participants.loc[included, "age"].min()), int(participants.loc[included, "age"].max())],
        "age_median": float(participants.loc[included, "age"].median()),
    }


def memory_versus_control(features):
    return with_holm(
        {f"load_{load}": paired_effect(features[f"mem_minus_ctrl_end_{load}"]) for load in LOADS}
    )


def effect_by_load(features, prefix):
    columns = [f"{prefix}_{load}" for load in LOADS]
    complete = features[columns].dropna()
    means = {
        f"load_{load}": {
            "mean": round(float(complete[f"{prefix}_{load}"].mean()), 4),
            "mean_ci95": [round(v, 4) for v in bootstrap_ci(complete[f"{prefix}_{load}"].to_numpy(), np.mean)],
        }
        for load in LOADS
    }
    pairwise = with_holm(
        {
            f"{high}_minus_{low}": paired_effect(complete[f"{prefix}_{high}"] - complete[f"{prefix}_{low}"])
            for low, high in LOAD_PAIRS
        }
    )
    return {
        "per_load": means,
        "friedman": friedman_effect([complete[column] for column in columns]),
        "pairwise": pairwise,
    }


def load_slope_tests(features):
    return {
        "end_of_sequence_slope_mm_per_digit": paired_effect(features["load_slope_end_mm_per_digit"]),
        "sequence_mean_slope_mm_per_digit": paired_effect(features["load_slope_sequence_mean_mm_per_digit"]),
        "quadratic_contrast_end": paired_effect(features["load_quadratic_contrast_end"]),
    }


def overload_tests(features):
    peak = features["peak_position_load13"].dropna()
    return {
        "slope_digits_1to7_vs_8to13": paired_effect(
            features["slope_digits_1to7_load13"] - features["slope_digits_8to13_load13"]
        ),
        "slope_digits_1to7_mean": round(float(features["slope_digits_1to7_load13"].mean()), 4),
        "slope_digits_8to13_mean": round(float(features["slope_digits_8to13_load13"].mean()), 4),
        "peak_position_median": float(peak.median()),
        "peak_position_iqr": [float(peak.quantile(0.25)), float(peak.quantile(0.75))],
    }


def tlx_between_person(features):
    predictors = {
        "load_slope_end": features["load_slope_end_mm_per_digit"],
        "mean_effect_end": features[[f"mem_minus_ctrl_end_{load}" for load in LOADS]].mean(axis=1),
    }
    results = {
        f"{name}_vs_tlx_{measure}": spearman_effect(values, features[f"tlx_{measure}_mean"])
        for name, values in predictors.items()
        for measure in TLX_MEASURES
    }
    return with_holm(results)


def tlx_within_person(features, administration_effects):
    long_tlx = pd.concat(
        [
            pd.DataFrame(
                {
                    "subject": features.index,
                    "tlx_administration": administration,
                    **{measure: features[f"tlx_{measure}_{administration}"].to_numpy() for measure in TLX_MEASURES},
                }
            )
            for administration in (1, 2, 3)
        ]
    )
    merged = administration_effects.merge(long_tlx, on=["subject", "tlx_administration"])
    results = {
        f"effect_vs_tlx_{measure}": repeated_measures_correlation(merged, "subject", "effect", measure)
        for measure in TLX_MEASURES
    }
    results = with_holm(results)
    results["mean_by_administration"] = {
        int(administration): {
            "effect_mm": round(float(group["effect"].mean()), 4),
            **{f"tlx_{measure}": round(float(group[measure].mean()), 2) for measure in TLX_MEASURES},
        }
        for administration, group in merged.groupby("tlx_administration")
    }
    return results


def split_half(trials, subjects, measure):
    odd, even = [], []
    for subject in subjects:
        memory = trials[(trials["subject"] == subject) & (trials["condition"] == "memory")].sort_values("trial")
        odd.append(measure(memory.iloc[0::2]))
        even.append(measure(memory.iloc[1::2]))
    correlation = spearman_effect(odd, even)
    rho = correlation["rho"]
    correlation["spearman_brown"] = round(2 * rho / (1 + rho), 3)
    return correlation, odd, even


def encoding_slope_of(trials):
    return linear_slope(range(1, 6), trials[ENCODING_COLUMNS].mean().to_numpy())


def reliability(trials, subjects):
    slope, odd, even = split_half(trials, subjects, encoding_slope_of)
    end9, _, _ = split_half(trials, subjects, lambda t: t.loc[t["load"] == 9, "d_end"].mean())
    return {"encoding_slope_memory": slope, "end_dilation_memory_load9": end9}, odd, even


def mixed_model(trials):
    data = trials.dropna(subset=["d_end", "baseline_mm"])
    fit = smf.mixedlm(
        "d_end ~ C(condition, Treatment('control')) * C(load) + baseline_mm",
        data,
        groups="subject",
    ).fit(reml=True)
    return {
        name: {"coef": round(float(fit.params[name]), 4), "p": float(fit.pvalues[name])}
        for name in fit.params.index
        if name != "Group Var"
    }


def exploratory_capacity(features):
    return {
        "capacity13_vs_peak_position": spearman_effect(
            features["capacity_partial_score_13"], features["peak_position_load13"]
        ),
        "capacity13_vs_mem_minus_ctrl_end_9": spearman_effect(
            features["capacity_partial_score_13"], features["mem_minus_ctrl_end_9"]
        ),
    }


def build_results():
    quality, trials, features, administration_effects = load_derived()
    subjects = features.index.tolist()
    reliability_results, odd, even = reliability(trials, subjects)
    results = {
        "sample": sample_description(quality, trials, features),
        "memory_vs_control_end_of_sequence": memory_versus_control(features),
        "load_effect_end_of_sequence": effect_by_load(features, "mem_minus_ctrl_end"),
        "load_effect_sequence_mean": effect_by_load(features, "mem_minus_ctrl_sequence_mean"),
        "within_sequence_slope_by_load": effect_by_load(features, "sequence_slope_mem_minus_ctrl"),
        "load_slope": load_slope_tests(features),
        "encoding_slope_digits_1to5": {
            "memory_vs_control": paired_effect(features["encoding_slope_memory"] - features["encoding_slope_control"]),
            "memory_vs_zero": paired_effect(features["encoding_slope_memory"]),
        },
        "overload_13_digits": overload_tests(features),
        "tlx_between_person": tlx_between_person(features),
        "tlx_within_person": tlx_within_person(features, administration_effects),
        "reliability": reliability_results,
        "mixed_model": mixed_model(trials),
        "exploratory_capacity": exploratory_capacity(features),
    }
    return results, trials, features, (odd, even)


def write_results(results):
    DS003838_RESULTS.mkdir(parents=True, exist_ok=True)
    path = DS003838_RESULTS / "stats.json"
    path.write_text(json.dumps(results, indent=2, default=float))
    return path


def main(argv=None):
    argparse.ArgumentParser(prog="pupilad validate").parse_args(argv)
    results, trials, features, split_halves = build_results()
    path = write_results(results)
    traces = load_ds003838_traces(features.index.tolist())
    plot_load_dilation(trials, traces, features, results, DS003838_RESULTS / "fig1_load_dilation.png")
    plot_reliability(split_halves, features, results, DS003838_RESULTS / "fig2_reliability_overload.png")
    plot_load_slope_and_tlx(features, results, DS003838_RESULTS / "fig3_load_slope_tlx.png")
    summary = {key: results[key] for key in ("sample", "load_slope", "tlx_between_person")}
    print(json.dumps(summary, indent=1, default=float))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
