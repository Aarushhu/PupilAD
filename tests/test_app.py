import json

import numpy as np
import pandas as pd
import pytest

flask = pytest.importorskip("flask")
pytest.importorskip("cv2")

from pupilad.app.camera import clock
from pupilad.app.server import TASK_COLUMNS, binned_trace, create_app
from pupilad.webcam.trial_table import build_trial_table


@pytest.fixture
def client(tmp_path):
    app = create_app(tmp_path, camera_index=0)
    with app.test_client() as test_client:
        test_client.logs_dir = tmp_path
        yield test_client


def nback_rows(start):
    letters = list("KMKTMKAB")
    rows = []
    for index, letter in enumerate(letters):
        is_match = index >= 2 and letter == letters[index - 2]
        responded = index in (2, 4)
        rows.append(
            {
                "trial": index,
                "letter": letter,
                "is_match": is_match,
                "responded": responded,
                "rt": 0.41 if responded else None,
                "correct": responded == is_match,
                "onset_time": start + 2.0 * index,
            }
        )
    return rows


def stroop_rows(start):
    trials = [("red", "red", "r", 0.5), ("red", "blue", "b", 0.7), ("green", "yellow", None, None)]
    return [
        {
            "trial": index,
            "word": word,
            "ink_color": ink,
            "is_congruent": word == ink,
            "responded": key is not None,
            "response_color": {"r": "red", "b": "blue"}.get(key),
            "rt": rt,
            "correct": {"r": "red", "b": "blue"}.get(key) == ink,
            "onset_time": start + 20 + index * 3.5,
        }
        for index, (word, ink, key, rt) in enumerate(trials)
    ]


def digit_rows(start):
    return [
        {
            "direction": "forward",
            "length": 3,
            "digits": "047",
            "response": "047",
            "correct": True,
            "stim_onset_time": start + 40,
            "recall_end_time": start + 46,
        },
        {
            "direction": "forward",
            "length": 4,
            "digits": "1234",
            "response": "1243",
            "correct": False,
            "stim_onset_time": start + 47,
            "recall_end_time": start + 54,
        },
        {
            "direction": "backward",
            "length": 3,
            "digits": "472",
            "response": "274",
            "correct": True,
            "stim_onset_time": start + 55,
            "recall_end_time": start + 61,
        },
    ]


def run_session(client, subject_id="T01"):
    start = clock()
    response = client.post(
        "/api/session/start",
        json={"subject_id": subject_id, "session": "1", "record_webcam": False, "tasks": []},
    )
    assert response.status_code == 200, response.json
    client.post("/api/session/marker", json={"label": "NBACK_START", "t": start})
    client.post("/api/session/results", json={"task": "nback", "rows": nback_rows(start)})
    client.post("/api/session/marker", json={"label": "NBACK_END", "t": start + 16})
    client.post("/api/session/results", json={"task": "stroop", "rows": stroop_rows(start)})
    client.post("/api/session/results", json={"task": "digitspan", "rows": digit_rows(start)})
    assert client.post("/api/session/finish", json={"aborted": False}).status_code == 200
    return start


def test_session_writes_battery_compatible_files(client):
    run_session(client)
    logs = client.logs_dir
    for task, columns in TASK_COLUMNS.items():
        frame = pd.read_csv(logs / f"T01_{task}.csv")
        assert list(frame.columns) == columns
    markers = pd.read_csv(logs / "T01_markers.csv")
    assert list(markers.columns) == ["wall_clock_time", "psychopy_clock_time", "event"]
    assert list(markers["event"]) == ["NBACK_START", "NBACK_END"]
    meta = json.loads((logs / "T01_session.json").read_text())
    assert meta["aborted"] is False
    assert sorted(meta["tasks_saved"]) == ["digitspan", "nback", "stroop"]


def test_existing_subject_requires_overwrite(client):
    run_session(client)
    response = client.post("/api/session/start", json={"subject_id": "T01", "record_webcam": False})
    assert response.status_code == 409
    response = client.post(
        "/api/session/start", json={"subject_id": "T01", "record_webcam": False, "overwrite": True}
    )
    assert response.status_code == 200
    assert not (client.logs_dir / "T01_nback.csv").exists()


def test_rejects_unsafe_subject_id(client):
    response = client.post(
        "/api/session/start", json={"subject_id": "../evil", "record_webcam": False}
    )
    assert response.status_code == 400


def test_summary_scores(client):
    run_session(client)
    summary = client.get("/api/sessions/T01").json["summary"]
    assert summary["nback"]["targets"] == 1
    assert summary["nback"]["hit_rate"] == 1.0
    assert summary["nback"]["false_alarm_rate"] == 0.2
    assert summary["nback"]["mean_hit_rt_ms"] == 410
    assert summary["digitspan"] == {"forward_span": 3, "backward_span": 3, "sequences": 3}
    assert summary["stroop"]["accuracy"] == pytest.approx(2 / 3, abs=1e-3)
    assert summary["stroop"]["interference_ms"] == 200
    listed = client.get("/api/sessions").json["sessions"]
    assert [row["subject_id"] for row in listed] == ["T01"]


def test_app_output_feeds_trial_table_and_trace(client):
    start = run_session(client)
    times = start - 2 + np.arange(0, 70, 1 / 30)
    diameters = 20 + np.sin(times)
    diameters[::7] = np.nan
    pd.DataFrame(
        {
            "subject_id": "T01",
            "frame_index": np.arange(len(times)),
            "wall_clock_time": "2026-11-03T10:00:00",
            "psychopy_clock_time": times,
            "pupil_diameter_px": diameters,
        }
    ).to_csv(client.logs_dir / "T01_pupil_trace.csv", index=False)

    table = build_trial_table(client.logs_dir, "T01")
    assert set(table["task_name"]) == {"nback", "stroop", "digit_span"}
    assert table.groupby("task_name")["trial_id"].nunique().to_dict() == {
        "digit_span": 3,
        "nback": 8,
        "stroop": 3,
    }

    trace = client.get("/api/sessions/T01/trace").json
    assert trace["frames"] == len(times)
    assert trace["intervals"][0]["task"] == "NBACK"
    assert trace["intervals"][0]["start"] == pytest.approx(2.0, abs=0.01)
    assert len(trace["t"]) <= 1500


def test_binned_trace_keeps_gaps():
    times = np.arange(10.0)
    values = np.array([1, 1, np.nan, np.nan, 2, 2, 2, 2, 3, 3], dtype=float)
    centers, medians = binned_trace(times, values, max_points=5)
    assert len(centers) == 5
    assert medians == [1.0, None, 2.0, 2.0, 3.0]


def test_marker_without_session_is_rejected(client):
    response = client.post("/api/session/marker", json={"label": "X", "t": 1.0})
    assert response.status_code == 409
