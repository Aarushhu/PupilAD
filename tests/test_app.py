import json
import time

import numpy as np
import pandas as pd
import pytest

flask = pytest.importorskip("flask")
cv2 = pytest.importorskip("cv2")

from pupilad.app.server import TASK_COLUMNS, binned_trace, create_app
from pupilad.webcam.trial_table import build_trial_table

START = 1234.5
EPOCH_MS = 1_793_000_000_000.0


@pytest.fixture
def client(tmp_path):
    app = create_app(tmp_path, auto_process=False)
    with app.test_client() as test_client:
        test_client.logs_dir = tmp_path
        yield test_client


def nback_rows(start):
    letters = list("KMKTMKAB")
    rows = []
    for index, letter in enumerate(letters):
        is_match = index >= 2 and letter == letters[index - 2]
        said_match = index in (2, 4)
        rows.append(
            {
                "trial": index,
                "letter": letter,
                "is_match": is_match,
                "responded": said_match,
                "rt": 0.41 if said_match else 0.62,
                "correct": said_match == is_match,
                "onset_time": start + 2.0 * index,
                "response": "match" if said_match else "no_match",
            }
        )
    return rows


def stroop_rows(start):
    trials = [("red", "red", "red", 0.5), ("red", "blue", "blue", 0.7), ("green", "yellow", None, None)]
    return [
        {
            "trial": index,
            "word": word,
            "ink_color": ink,
            "is_congruent": word == ink,
            "responded": answer is not None,
            "response_color": answer,
            "rt": rt,
            "correct": answer == ink,
            "onset_time": start + 20 + index * 3.5,
        }
        for index, (word, ink, answer, rt) in enumerate(trials)
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


def mark(client, label, t):
    return client.post(
        "/api/session/marker", json={"label": label, "t": t, "epoch_ms": EPOCH_MS + (t - START) * 1000}
    )


def start_session(client, subject_id="T01", **extra):
    body = {"subject_id": subject_id, "session": "1", "record_webcam": False, "tasks": [], **extra}
    return client.post("/api/session/start", json=body)


def run_session(client, subject_id="T01"):
    response = start_session(client, subject_id)
    assert response.status_code == 200, response.json
    mark(client, "NBACK_START", START)
    client.post("/api/session/results", json={"task": "nback", "rows": nback_rows(START)})
    mark(client, "NBACK_END", START + 16)
    client.post("/api/session/results", json={"task": "stroop", "rows": stroop_rows(START)})
    client.post("/api/session/results", json={"task": "digitspan", "rows": digit_rows(START)})
    assert client.post("/api/session/finish", json={"aborted": False}).status_code == 200
    return START


def make_video(path, frames=45, fps=30):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    for index in range(frames):
        frame = np.full((120, 160, 3), index * 4 % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path.read_bytes()


def test_session_writes_battery_compatible_files(client):
    run_session(client)
    logs = client.logs_dir
    for task, columns in TASK_COLUMNS.items():
        frame = pd.read_csv(logs / f"T01_{task}.csv")
        assert list(frame.columns) == columns
    markers = pd.read_csv(logs / "T01_markers.csv")
    assert list(markers.columns) == ["wall_clock_time", "psychopy_clock_time", "event"]
    assert list(markers["event"]) == ["NBACK_START", "NBACK_END"]
    assert markers["psychopy_clock_time"].tolist() == [START, START + 16]
    meta = json.loads((logs / "T01_session.json").read_text())
    assert meta["aborted"] is False
    assert sorted(meta["tasks_saved"]) == ["digitspan", "nback", "stroop"]


def test_existing_subject_requires_overwrite(client):
    run_session(client)
    assert start_session(client).status_code == 409
    assert start_session(client, overwrite=True).status_code == 200
    assert not (client.logs_dir / "T01_nback.csv").exists()


def test_rejects_unsafe_subject_id(client):
    assert start_session(client, "../evil").status_code == 400


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
    assert mark(client, "X", 1.0).status_code == 409


def test_uploaded_video_becomes_aligned_mp4_and_timestamps(client, tmp_path):
    data = make_video(tmp_path / "source.mp4")
    assert start_session(client, "V01", record_webcam=True).status_code == 200
    half = len(data) // 2
    for seq, part in enumerate([data[:half], data[half:]]):
        response = client.post(f"/api/session/media/video?seq={seq}&mime=video/mp4", data=part)
        assert response.status_code == 200, response.json
    assert client.post("/api/session/media/video?seq=0&mime=video/mp4", data=data[:half]).json["next"] == 2
    assert client.post("/api/session/media/video?seq=5&mime=video/mp4", data=b"x").status_code == 409
    recording = {"mime": "video/mp4", "start_s": START, "start_epoch_ms": EPOCH_MS, "duration_s": 1.5}
    client.post("/api/session/finish", json={"aborted": False, "recording": recording})
    assert (client.logs_dir / "V01_webcam_raw.mp4").read_bytes() == data

    client.post("/api/sessions/V01/analyze", json={})
    for _ in range(100):
        job = client.get("/api/sessions/V01/job").json
        if job["state"] != "running":
            break
        time.sleep(0.1)
    assert job["state"] == "done", job
    stamps = pd.read_csv(client.logs_dir / "V01_webcam_frame_timestamps.csv")
    assert list(stamps.columns) == ["frame_index", "wall_clock_time", "psychopy_clock_time"]
    assert len(stamps) == 45
    assert stamps["psychopy_clock_time"].iloc[0] == pytest.approx(START)
    assert stamps["psychopy_clock_time"].iloc[-1] == pytest.approx(START + 44 / 30, abs=0.002)
    assert (client.logs_dir / "V01_webcam.mp4").exists()
    assert (client.logs_dir / "V01_pupil_trace.csv").exists()
    meta = client.get("/api/sessions/V01").json["meta"]
    assert meta["webcam"]["frames"] == 45
    assert meta["pupil_detect_rate"] == 0.0


def test_audio_upload_is_saved_with_its_extension(client):
    start_session(client, "A01")
    client.post("/api/session/media/audio?seq=0&mime=audio/webm;codecs=opus", data=b"abc")
    client.post("/api/session/media/audio?seq=1&mime=audio/webm;codecs=opus", data=b"def")
    client.post("/api/session/finish", json={"aborted": False})
    assert (client.logs_dir / "A01_fluency_audio.webm").read_bytes() == b"abcdef"
    assert client.get("/api/sessions/A01").json["files"]["_fluency_audio.webm"] is True


def test_camera_check_reports_no_face_on_blank_frame(client):
    ok, jpeg = cv2.imencode(".jpg", np.full((240, 320, 3), 128, dtype=np.uint8))
    result = client.post("/api/camera/check", data=jpeg.tobytes()).json
    assert result == {"width": 320, "height": 240, "face": None, "eyes": 0, "pupil_px": None}
    assert client.post("/api/camera/check", data=b"not an image").status_code == 400


def test_lan_access_requires_key(tmp_path):
    app = create_app(tmp_path, access_key="secret123", lan_url="https://10.0.0.5:5050/?key=secret123")
    remote = {"REMOTE_ADDR": "10.0.0.9"}
    with app.test_client() as test_client:
        assert test_client.get("/api/info", environ_base=remote).status_code == 403
        response = test_client.get("/?key=secret123", environ_base=remote)
        assert response.status_code == 302
        assert "pupilad_key=secret123" in response.headers["Set-Cookie"]
        assert test_client.get("/api/info", environ_base=remote).json["lan_url"] is None
    with app.test_client() as local_client:
        assert local_client.get("/api/info").json["lan_url"].startswith("https://10.0.0.5")
