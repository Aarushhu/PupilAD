import math

import pandas as pd

NBACK_N = 2


def number(value, digits=3):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(value) else round(value, digits)


def as_bool(series):
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])


def read_log(path, text_columns=()):
    if not path.exists():
        return None
    frame = pd.read_csv(path, dtype={column: str for column in text_columns})
    return frame if len(frame) else None


def nback_summary(frame, n=NBACK_N):
    is_match = as_bool(frame["is_match"])
    responded = as_bool(frame["responded"])
    scorable = frame["trial"] >= n
    matches = scorable & is_match
    non_matches = scorable & ~is_match
    hits = matches & responded
    return {
        "trials": int(len(frame)),
        "accuracy": number(as_bool(frame["correct"]).mean()),
        "targets": int(matches.sum()),
        "hit_rate": number(responded[matches].mean()) if matches.any() else None,
        "false_alarm_rate": number(responded[non_matches].mean()) if non_matches.any() else None,
        "mean_hit_rt_ms": number(frame.loc[hits, "rt"].mean() * 1000, 0) if hits.any() else None,
    }


def longest_correct(frame, direction):
    rows = frame[(frame["direction"] == direction) & as_bool(frame["correct"])]
    return int(rows["length"].max()) if len(rows) else 0


def digit_span_summary(frame):
    return {
        "forward_span": longest_correct(frame, "forward"),
        "backward_span": longest_correct(frame, "backward"),
        "sequences": int(len(frame)),
    }


def stroop_summary(frame):
    correct = as_bool(frame["correct"])
    congruent = as_bool(frame["is_congruent"])
    responded = as_bool(frame["responded"])
    rt = pd.to_numeric(frame["rt"], errors="coerce") * 1000
    congruent_rt = rt[correct & congruent].mean()
    incongruent_rt = rt[correct & ~congruent].mean()
    return {
        "trials": int(len(frame)),
        "accuracy": number(correct.mean()),
        "missed": int((~responded).sum()),
        "congruent_rt_ms": number(congruent_rt, 0),
        "incongruent_rt_ms": number(incongruent_rt, 0),
        "interference_ms": number(incongruent_rt - congruent_rt, 0),
    }


def session_summary(logs_dir, subject_id):
    summary = {}
    nback = read_log(logs_dir / f"{subject_id}_nback.csv", ["letter"])
    if nback is not None:
        summary["nback"] = nback_summary(nback)
    digit_span = read_log(logs_dir / f"{subject_id}_digitspan.csv", ["digits", "response"])
    if digit_span is not None:
        summary["digitspan"] = digit_span_summary(digit_span)
    stroop = read_log(logs_dir / f"{subject_id}_stroop.csv")
    if stroop is not None:
        summary["stroop"] = stroop_summary(stroop)
    return summary


def task_intervals(markers):
    starts = {}
    intervals = []
    for _, row in markers.iterrows():
        label = str(row["event"])
        if label.endswith("_START"):
            starts[label[: -len("_START")]] = row["psychopy_clock_time"]
        elif label.endswith("_END"):
            name = label[: -len("_END")]
            if name in starts:
                intervals.append((name, starts.pop(name), row["psychopy_clock_time"]))
    return intervals
