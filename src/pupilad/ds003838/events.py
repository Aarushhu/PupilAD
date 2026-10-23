import numpy as np
import pandas as pd

MEMORY_PREFIX = "6"
CONTROL_PREFIX = "5"


def parse_event_label(label):
    code = str(int(label))
    if len(code) == 7 and code.startswith(MEMORY_PREFIX):
        return {
            "condition": "memory",
            "position": int(code[1:4]),
            "load": int(code[4:6]),
            "recalled": int(code[6]),
        }
    if len(code) == 6 and code.startswith(CONTROL_PREFIX):
        return {
            "condition": "control",
            "position": int(code[1:4]),
            "load": int(code[4:6]),
            "recalled": np.nan,
        }
    return None


def digit_events(events):
    rows = []
    for timestamp, label in zip(events["timestamp"], events["label"]):
        parsed = parse_event_label(label)
        if parsed is not None:
            rows.append({"time": float(timestamp), **parsed})
    digits = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    digits["trial"] = (digits["position"] == 1).cumsum()
    return digits


def is_complete_sequence(trial_digits):
    load = int(trial_digits["load"].iloc[0])
    expected_positions = np.arange(1, load + 1)
    return len(trial_digits) == load and np.array_equal(
        trial_digits["position"].to_numpy(), expected_positions
    )
