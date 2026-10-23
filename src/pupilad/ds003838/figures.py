import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from pupilad.ds003838.features import LOADS
from pupilad.ds003838.preprocess import BASELINE_S, OUTPUT_RATE_HZ, TRACE_LENGTH

LOAD_COLORS = {5: "#86b6ef", 9: "#2a78d6", 13: "#0d366b"}
INK = "#1f1f1f"
MUTED = "#6b6b6b"
GRID = "#e6e6e6"
CONTROL_DASH = (0, (4, 3))

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def style_axis(axis, horizontal_grid_only=True):
    axis.grid(axis="y" if horizontal_grid_only else "both", color=GRID, lw=0.8)
    axis.set_axisbelow(True)


def panel_title(axis, text):
    axis.set_title(text, loc="left", fontsize=11, color=INK, fontweight="bold")


def condition_average_trace(trials, traces, condition, load):
    per_subject = []
    for subject, subject_traces in traces.items():
        trial_numbers = trials.loc[
            (trials["subject"] == subject)
            & (trials["condition"] == condition)
            & (trials["load"] == load),
            "trial",
        ].to_numpy() - 1
        trial_numbers = trial_numbers[trial_numbers < len(subject_traces)]
        if len(trial_numbers):
            per_subject.append(np.nanmean(subject_traces[trial_numbers], axis=0))
    per_subject = np.array(per_subject)
    mean = np.nanmean(per_subject, axis=0)
    standard_error = np.nanstd(per_subject, axis=0) / np.sqrt(np.isfinite(per_subject).sum(axis=0))
    return mean, standard_error


def draw_time_courses(axis, trials, traces):
    seconds = np.arange(TRACE_LENGTH) / OUTPUT_RATE_HZ - BASELINE_S
    for load in LOADS:
        end = int((2 * load + 2 + BASELINE_S) * OUTPUT_RATE_HZ)
        for condition, line_style in (("control", CONTROL_DASH), ("memory", "-")):
            mean, error = condition_average_trace(trials, traces, condition, load)
            axis.plot(seconds[:end], mean[:end], color=LOAD_COLORS[load], lw=2, ls=line_style)
            axis.fill_between(
                seconds[:end], mean[:end] - error[:end], mean[:end] + error[:end],
                color=LOAD_COLORS[load], alpha=0.15, lw=0,
            )
            if condition == "memory":
                axis.annotate(
                    f"{load} digits", (seconds[end - 1], mean[end - 1]), xytext=(4, 0),
                    textcoords="offset points", color=INK, va="center", fontsize=9,
                )
    axis.axhline(0, color=MUTED, lw=0.8)
    style_axis(axis)
    axis.set_xlim(-0.5, 31)
    axis.set_xlabel("Time from first digit (s) · one digit every 2 s")
    axis.set_ylabel("Pupil change from baseline (mm)")
    axis.legend(
        handles=[
            Line2D([], [], color=INK, lw=2, label="Memorize"),
            Line2D([], [], color=INK, lw=2, ls=CONTROL_DASH, label="Just listen"),
        ],
        frameon=False, loc="upper left",
    )


def draw_points_by_load(axis, values_by_load, ylabel, ci_by_load):
    rng = np.random.default_rng(0)
    for index, load in enumerate(LOADS):
        values = values_by_load[load].dropna().to_numpy()
        jitter = index + rng.uniform(-0.12, 0.12, len(values))
        axis.scatter(jitter, values, s=14, color=LOAD_COLORS[load], alpha=0.5, lw=0)
        low, high = ci_by_load[load]
        mean = values.mean()
        axis.errorbar(
            index, mean, yerr=[[mean - low], [high - mean]], fmt="o", color=INK,
            ms=7, capsize=5, lw=1.6, zorder=5,
        )
    axis.plot(range(len(LOADS)), [values_by_load[load].mean() for load in LOADS], color=INK, lw=1.2)
    axis.axhline(0, color=MUTED, lw=0.8)
    style_axis(axis)
    axis.set_xticks(range(len(LOADS)), [str(load) for load in LOADS])
    axis.set_xlabel("Digits in the sequence")
    axis.set_ylabel(ylabel)


def plot_load_dilation(trials, traces, features, results, path):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [2.3, 1]})
    draw_time_courses(axes[0], trials, traces)
    panel_title(axes[0], "A  Memorizing keeps the pupil larger than listening")
    load_effect = results["load_effect_end_of_sequence"]
    draw_points_by_load(
        axes[1],
        {load: features[f"mem_minus_ctrl_end_{load}"] for load in LOADS},
        "Memorize − listen, last digit (mm)",
        {load: load_effect["per_load"][f"load_{load}"]["mean_ci95"] for load in LOADS},
    )
    friedman = load_effect["friedman"]
    panel_title(axes[1], "B  Effort effect by load")
    axes[1].text(
        0.02, 0.98,
        f"Friedman χ²={friedman['chi2']:.1f}, p={friedman['p']:.1g}\nmean, bootstrap 95% CI; dots = people",
        transform=axes[1].transAxes, va="top", fontsize=8.5, color=MUTED,
    )
    n_included = results["sample"]["n_included"]
    figure.suptitle(
        f"OpenNeuro ds003838 · n = {n_included} young adults · Pupil Labs Core, 120 Hz",
        x=0.01, ha="left", fontsize=9, color=MUTED,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def plot_reliability(split_halves, features, results, path):
    odd, even = split_halves
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].scatter(odd, even, s=22, color=LOAD_COLORS[9], alpha=0.7, lw=0)
    limits = [np.nanmin(odd + even), np.nanmax(odd + even)]
    axes[0].plot(limits, limits, color=MUTED, lw=0.8, ls=CONTROL_DASH)
    style_axis(axes[0], horizontal_grid_only=False)
    axes[0].set_xlabel("Encoding slope, odd trials (mm/digit)")
    axes[0].set_ylabel("Encoding slope, even trials (mm/digit)")
    slope_reliability = results["reliability"]["encoding_slope_memory"]
    panel_title(
        axes[0],
        f"A  Split-half ρ = {slope_reliability['rho']:.2f} (Spearman–Brown {slope_reliability['spearman_brown']:.2f})",
    )
    early = features["slope_digits_1to7_load13"]
    late = features["slope_digits_8to13_load13"]
    for early_value, late_value in zip(early, late):
        axes[1].plot([0, 1], [early_value, late_value], color=LOAD_COLORS[13], alpha=0.25, lw=1)
    axes[1].plot([0, 1], [early.mean(), late.mean()], color=INK, lw=2, marker="o", ms=8)
    axes[1].axhline(0, color=MUTED, lw=0.8)
    style_axis(axes[1])
    axes[1].set_xticks([0, 1], ["Digits 1–7", "Digits 8–13"])
    axes[1].set_ylabel("Pupil slope, 13-digit memory trials (mm/digit)")
    panel_title(axes[1], "B  Past ~7 digits the pupil stops rising")
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def plot_load_slope_and_tlx(features, results, path):
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    slopes = features["load_slope_end_mm_per_digit"].dropna()
    rng = np.random.default_rng(1)
    axes[0].scatter(rng.uniform(-0.15, 0.15, len(slopes)), slopes, s=18, color=LOAD_COLORS[9], alpha=0.6, lw=0)
    slope_test = results["load_slope"]["end_of_sequence_slope_mm_per_digit"]
    low, high = slope_test["mean_ci95"]
    axes[0].errorbar(0, slope_test["mean"], yerr=[[slope_test["mean"] - low], [high - slope_test["mean"]]],
                     fmt="o", color=INK, ms=8, capsize=6, lw=1.6, zorder=5)
    axes[0].axhline(0, color=MUTED, lw=0.8)
    axes[0].set_xlim(-0.6, 0.6)
    axes[0].set_xticks([])
    style_axis(axes[0])
    axes[0].set_ylabel("Linear load slope, 5→9→13 digits (mm per digit)")
    panel_title(axes[0], f"A  Load slope  mean {slope_test['mean']:.3f} [{low:.3f}, {high:.3f}]")
    tlx = features["tlx_mental_mean"]
    axes[1].scatter(tlx, features["load_slope_end_mm_per_digit"], s=22, color=LOAD_COLORS[13], alpha=0.7, lw=0)
    style_axis(axes[1], horizontal_grid_only=False)
    axes[1].set_xlabel("NASA-TLX mental demand (mean of 3 ratings)")
    axes[1].set_ylabel("Load slope (mm per digit)")
    correlation = results["tlx_between_person"]["load_slope_end_vs_tlx_mental"]
    rho_low, rho_high = correlation["rho_ci95"]
    panel_title(axes[1], f"B  vs NASA-TLX  ρ = {correlation['rho']:.2f} [{rho_low:.2f}, {rho_high:.2f}]")
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)
