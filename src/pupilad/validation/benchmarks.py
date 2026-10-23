PLAUSIBLE_PUPIL_RANGE_MM = (1.5, 9.0)

LITERATURE_BENCHMARKS = {
    "plausible_pupil_diameter_mm": {
        "range": PLAUSIBLE_PUPIL_RANGE_MM,
        "type": "sanity_range",
        "note": (
            "Physiologically plausible human pupil diameter under typical lab lighting. "
            "Values outside this range usually indicate a tracking artifact."
        ),
    },
    "constriction_velocity_cutoff_high": {
        "value_mm_s": 4.43,
        "type": "published_cutoff",
        "citation": "UNVERIFIED: constriction-velocity cutoff study, source not recorded",
        "verified": False,
    },
    "constriction_velocity_cutoff_low": {
        "value_mm_s": 3.12,
        "type": "published_cutoff",
        "citation": "UNVERIFIED: constriction-velocity cutoff study, source not recorded",
        "verified": False,
    },
    "pupil_size_change_cutoff": {
        "value_mm": 1.77,
        "type": "published_cutoff",
        "citation": "UNVERIFIED: pupil-size-change cutoff study, source not recorded",
        "verified": False,
        "reported_at_cutoff": {
            "sensitivity": 0.613,
            "specificity": 0.80,
            "PPV": 0.4776,
            "NPV": 0.8688,
        },
    },
    "dilation_amplitude_vs_cognitive_load": {
        "type": "direction_only",
        "citation": "Granholm et al. 2017, J Alzheimers Dis 56(4):1419-1428, doi:10.3233/JAD-161078",
        "verified": True,
        "expected_direction": (
            "Amnestic single-domain MCI participants showed greater dilation than cognitively "
            "normal participants at 3- and 6-digit loads despite normal accuracy."
        ),
    },
    "dilation_amplitude_tau_association": {
        "type": "direction_only",
        "citation": "UNVERIFIED: Granholm et al. 2021 (tau rather than amyloid association)",
        "verified": False,
        "expected_direction": "Pupillary dilation response relates to tau burden rather than amyloid burden.",
    },
}

ADNI_CSF_REFERENCE = {
    "source": "Local ADNI extract: UPENNBIOMK_MASTER.csv joined to DXSUM.csv on (RID, VISCODE)",
    "n_visits": 3530,
    "computed": "2026-09-20",
    "markers": {
        "PTAU": {
            "CN": {"mean": 28.546, "std": 16.412, "median": 23.0, "n": 1055},
            "MCI": {"mean": 36.529, "std": 20.943, "median": 32.0, "n": 1604},
            "Dementia": {"mean": 43.644, "std": 20.437, "median": 38.6, "n": 847},
            "cohens_d_cn_vs_mci": 0.414,
            "cohens_d_cn_vs_dementia": 0.824,
            "p_cn_vs_mci": 1.11e-30,
            "p_cn_vs_dementia": 9.67e-87,
        },
        "TAU": {
            "CN": {"mean": 73.577, "std": 30.511, "median": 67.60, "n": 1047},
            "MCI": {"mean": 98.504, "std": 56.057, "median": 83.15, "n": 1606},
            "Dementia": {"mean": 120.725, "std": 61.033, "median": 109.0, "n": 849},
            "cohens_d_cn_vs_mci": 0.523,
            "cohens_d_cn_vs_dementia": 1.009,
            "p_cn_vs_mci": 1.77e-29,
            "p_cn_vs_dementia": 1.64e-91,
        },
        "ABETA": {
            "CN": {"mean": 205.614, "std": 55.522, "median": 215.5, "n": 1058},
            "MCI": {"mean": 166.924, "std": 51.795, "median": 150.0, "n": 1602},
            "Dementia": {"mean": 139.423, "std": 36.032, "median": 136.0, "n": 852},
            "cohens_d_cn_vs_mci": -0.726,
            "cohens_d_cn_vs_dementia": -1.384,
            "p_cn_vs_mci": 6.96e-65,
            "p_cn_vs_dementia": 1.70e-134,
        },
    },
}
