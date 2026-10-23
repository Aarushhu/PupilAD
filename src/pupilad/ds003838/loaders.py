import json

import numpy as np
import pandas as pd

from pupilad.paths import DS003838_DERIVED, DS003838_RAW, DS003838_SLIM


def load_ds003838_participants():
    return pd.read_csv(DS003838_RAW / "participants.tsv", sep="\t").set_index("participant_id")


def load_ds003838_samples(subject):
    return pd.read_parquet(DS003838_SLIM / f"{subject}.parquet")


def load_ds003838_events(subject):
    return pd.read_csv(
        DS003838_RAW / subject / "pupil" / f"{subject}_task-memory_events.tsv", sep="\t"
    )


def load_ds003838_behaviour(subjects):
    tables = [
        pd.read_csv(path, sep="\t")
        for path in (
            DS003838_RAW / subject / "beh" / f"{subject}_task-memory_beh.tsv"
            for subject in subjects
        )
        if path.exists()
    ]
    return pd.concat(tables, ignore_index=True)


def subjects_with_samples():
    return sorted(path.stem for path in DS003838_SLIM.glob("sub-*.parquet"))


def load_ds003838_quality():
    summaries = [
        json.loads(path.read_text())
        for path in sorted((DS003838_DERIVED / "trials").glob("*_qc.json"))
    ]
    return pd.DataFrame(summaries).sort_values("subject").reset_index(drop=True)


def load_ds003838_trials(subjects):
    return pd.concat(
        [
            pd.read_csv(DS003838_DERIVED / "trials" / f"{subject}_trials.csv")
            for subject in subjects
        ],
        ignore_index=True,
    )


def load_ds003838_traces(subjects):
    return {
        subject: np.load(DS003838_DERIVED / "trials" / f"{subject}_traces.npy")
        for subject in subjects
    }
