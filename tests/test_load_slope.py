import numpy as np
import pandas as pd
import pytest

from pupilad.ds003838.features import (
    LOADS,
    digit_slope,
    memory_minus_control,
    quadratic_contrast,
    slope_across_loads,
)
from pupilad.model.lstm import score_to_tier
from pupilad.stats import holm_adjust, paired_effect, repeated_measures_correlation


def cells_with_end_values(memory_end, control_end):
    rows = []
    for load in LOADS:
        rows.append(
            {"subject": "sub-1", "condition": "memory", "load": load, "d_end": memory_end[load]}
        )
        rows.append(
            {"subject": "sub-1", "condition": "control", "load": load, "d_end": control_end[load]}
        )
    return pd.DataFrame(rows)


def test_memory_minus_control_by_load():
    cells = cells_with_end_values({5: 0.3, 9: 0.5, 13: 0.4}, {5: 0.1, 9: 0.1, 13: 0.1})
    effect = memory_minus_control(cells)
    assert effect.loc["sub-1"].round(6).tolist() == [0.2, 0.4, 0.3]


def test_slope_across_loads_is_mm_per_digit():
    effect = pd.DataFrame({5: [0.1], 9: [0.3], 13: [0.5]}, index=["sub-1"])
    assert slope_across_loads(effect).iloc[0] == pytest.approx(0.05)


def test_quadratic_contrast_is_negative_for_inverted_u():
    inverted_u = pd.DataFrame({5: [0.2], 9: [0.4], 13: [0.2]})
    linear = pd.DataFrame({5: [0.1], 9: [0.3], 13: [0.5]})
    assert quadratic_contrast(inverted_u).iloc[0] == pytest.approx(-0.4)
    assert quadratic_contrast(linear).iloc[0] == pytest.approx(0.0)


def test_digit_slope_within_sequence():
    row = {"subject": "sub-1", "condition": "memory", "load": 13}
    row.update({f"d{digit}": 0.02 * digit for digit in range(1, 15)})
    slope = digit_slope(pd.DataFrame([row]), "memory", 13, tuple(range(1, 8)))
    assert slope.iloc[0] == pytest.approx(0.02)


def test_paired_effect_reports_effect_size_and_interval():
    differences = np.random.default_rng(0).normal(0.3, 0.3, 40)
    result = paired_effect(differences)
    low, high = result["mean_ci95"]
    assert low < result["mean"] < high
    assert result["dz"] == pytest.approx(differences.mean() / differences.std(ddof=1), abs=1e-3)
    assert result["p"] < 0.001


def test_holm_adjustment_matches_hand_calculation():
    assert holm_adjust([0.01, 0.04, 0.03]).round(4).tolist() == [0.03, 0.06, 0.06]


def test_repeated_measures_correlation_ignores_between_subject_offsets():
    rows = []
    for subject, offset in enumerate([0, 10, 20]):
        for step in range(3):
            rows.append({"subject": subject, "x": offset + step, "y": -offset + 2 * step})
    result = repeated_measures_correlation(pd.DataFrame(rows), "subject", "x", "y")
    assert result["r_rm"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "score, tier",
    [(85, "normal"), (70, "normal"), (69.5, "watch"), (50, "watch"), (49.9, "MCI-likely")],
)
def test_score_to_tier_has_no_gaps(score, tier):
    assert score_to_tier(score) == tier
