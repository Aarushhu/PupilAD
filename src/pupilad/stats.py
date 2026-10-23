import numpy as np
import pandas as pd
from scipy import stats

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 2027


def finite(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def cohens_dz(differences):
    differences = finite(differences)
    return float(differences.mean() / differences.std(ddof=1))


def bootstrap_ci(values, statistic, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    estimates = np.array([statistic(values[row]) for row in indices])
    low, high = np.nanpercentile(estimates, [2.5, 97.5])
    return [float(low), float(high)]


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    if np.isnan(p_values).any():
        return np.full(len(p_values), np.nan)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running_max)
    return adjusted


def rank_biserial(differences):
    differences = finite(differences)
    nonzero = differences[differences != 0]
    ranks = stats.rankdata(np.abs(nonzero))
    total = len(nonzero) * (len(nonzero) + 1) / 2
    return float((ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / total)


def paired_effect(differences):
    differences = finite(differences)
    test = stats.wilcoxon(differences)
    return {
        "n": int(len(differences)),
        "mean": round(float(differences.mean()), 4),
        "mean_ci95": [round(v, 4) for v in bootstrap_ci(differences, np.mean)],
        "median": round(float(np.median(differences)), 4),
        "dz": round(cohens_dz(differences), 3),
        "dz_ci95": [round(v, 3) for v in bootstrap_ci(differences, cohens_dz)],
        "rank_biserial": round(rank_biserial(differences), 3),
        "wilcoxon_W": float(test.statistic),
        "p": float(test.pvalue),
    }


def spearman_effect(x, y):
    pairs = pd.concat([pd.Series(x), pd.Series(y)], axis=1).dropna().to_numpy()

    def rho(sample):
        return stats.spearmanr(sample[:, 0], sample[:, 1]).statistic

    result = stats.spearmanr(pairs[:, 0], pairs[:, 1])
    return {
        "n": int(len(pairs)),
        "rho": round(float(result.statistic), 3),
        "rho_ci95": [round(v, 3) for v in bootstrap_ci(pairs, rho)],
        "p": float(result.pvalue),
    }


def repeated_measures_correlation(table, subject_column, x_column, y_column):
    table = table.dropna(subset=[x_column, y_column])
    centered_x = table[x_column] - table.groupby(subject_column)[x_column].transform("mean")
    centered_y = table[y_column] - table.groupby(subject_column)[y_column].transform("mean")
    r = float(np.corrcoef(centered_x, centered_y)[0, 1])
    n_subjects = table[subject_column].nunique()
    degrees_of_freedom = len(table) - n_subjects - 1
    p = (
        0.0
        if abs(r) >= 1
        else float(
            2 * stats.t.sf(abs(r) * np.sqrt(degrees_of_freedom / (1 - r**2)), degrees_of_freedom)
        )
    )
    z_half_width = 1.96 / np.sqrt(degrees_of_freedom - 1)
    ci = np.tanh(np.arctanh(r) + np.array([-z_half_width, z_half_width]))
    return {
        "n_subjects": int(n_subjects),
        "n_observations": int(len(table)),
        "df": int(degrees_of_freedom),
        "r_rm": round(r, 3),
        "r_rm_ci95": [round(float(v), 3) for v in ci],
        "p": p,
    }


def friedman_effect(columns):
    result = stats.friedmanchisquare(*columns)
    n_subjects = len(columns[0])
    return {
        "n": int(n_subjects),
        "chi2": round(float(result.statistic), 3),
        "p": float(result.pvalue),
        "kendalls_W": round(float(result.statistic / (n_subjects * (len(columns) - 1))), 3),
    }
