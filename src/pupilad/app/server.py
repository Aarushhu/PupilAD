import argparse
import csv
import json
import re
import threading
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, Response, abort, jsonify, request, send_from_directory

from pupilad.app import scoring
from pupilad.app.camera import Camera, clock
from pupilad.paths import DATA_DIR

DEFAULT_LOGS_DIR = DATA_DIR / "task_battery_logs"
STATIC_DIR = Path(__file__).with_name("static")
SUBJECT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
APP_VERSION = "1"
MAX_TRACE_POINTS = 1500

TASK_COLUMNS = {
    "nback": ["trial", "letter", "is_match", "responded", "rt", "correct", "onset_time"],
    "digitspan": [
        "direction",
        "length",
        "digits",
        "response",
        "correct",
        "stim_onset_time",
        "recall_end_time",
    ],
    "stroop": [
        "trial",
        "word",
        "ink_color",
        "is_congruent",
        "responded",
        "response_color",
        "rt",
        "correct",
        "onset_time",
    ],
}
SESSION_FILE_SUFFIXES = [
    "_session.json",
    "_markers.csv",
    "_nback.csv",
    "_digitspan.csv",
    "_stroop.csv",
    "_webcam.mp4",
    "_webcam_frame_timestamps.csv",
    "_pupil_trace.csv",
    "_trial_pupil_table.csv",
]


class Session:
    def __init__(self, logs_dir, subject_id, meta, camera, record_webcam):
        self.logs_dir = logs_dir
        self.subject_id = subject_id
        self.meta = meta
        self.camera = camera if record_webcam else None
        self.lock = threading.Lock()
        self.finished = False
        logs_dir.mkdir(parents=True, exist_ok=True)
        if self.camera is not None:
            self.camera.start_recording(
                self.path("_webcam.mp4"), self.path("_webcam_frame_timestamps.csv")
            )
        self.marker_handle = open(self.path("_markers.csv"), "w", newline="")
        self.marker_writer = csv.writer(self.marker_handle)
        self.marker_writer.writerow(["wall_clock_time", "psychopy_clock_time", "event"])
        self.write_meta()

    def path(self, suffix):
        return self.logs_dir / f"{self.subject_id}{suffix}"

    def write_meta(self):
        self.path("_session.json").write_text(json.dumps(self.meta, indent=2))

    def marker(self, label, timestamp):
        wall = datetime.now() - timedelta(seconds=clock() - timestamp)
        with self.lock:
            if self.finished:
                return
            self.marker_writer.writerow([wall.isoformat(), timestamp, label])
            self.marker_handle.flush()

    def save_task(self, task, rows):
        frame = pd.DataFrame(rows, columns=TASK_COLUMNS[task])
        frame.to_csv(self.path(f"_{task}.csv"), index=False)
        with self.lock:
            self.meta.setdefault("tasks_saved", [])
            if task not in self.meta["tasks_saved"]:
                self.meta["tasks_saved"].append(task)
            self.write_meta()

    def finish(self, aborted, extra):
        with self.lock:
            if self.finished:
                return self.meta
            self.finished = True
            self.marker_handle.close()
        self.meta["webcam"] = self.camera.stop_recording() if self.camera is not None else None
        self.meta["ended_at"] = datetime.now().isoformat()
        self.meta["aborted"] = bool(aborted)
        self.meta.update(extra or {})
        self.write_meta()
        return self.meta


def session_files(logs_dir, subject_id):
    return {
        suffix: (logs_dir / f"{subject_id}{suffix}").exists() for suffix in SESSION_FILE_SUFFIXES
    }


def known_subject_ids(logs_dir):
    if not logs_dir.exists():
        return []
    ids = set()
    for suffix in ("_markers.csv", "_session.json"):
        ids.update(path.name[: -len(suffix)] for path in logs_dir.glob(f"*{suffix}"))
    return sorted(ids)


def read_meta(logs_dir, subject_id):
    path = logs_dir / f"{subject_id}_session.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def binned_trace(times, values, max_points=MAX_TRACE_POINTS):
    if len(times) == 0:
        return [], []
    bins = min(max_points, len(times))
    edges = np.linspace(times[0], times[-1] + 1e-9, bins + 1)
    index = np.clip(np.searchsorted(edges, times, side="right") - 1, 0, bins - 1)
    frame = pd.DataFrame({"bin": index, "value": values})
    medians = frame.groupby("bin")["value"].median().reindex(range(bins))
    centers = (edges[:-1] + edges[1:]) / 2
    return centers.round(3).tolist(), [
        None if pd.isna(value) else round(float(value), 3) for value in medians
    ]


def create_app(logs_dir=DEFAULT_LOGS_DIR, camera_index=0):
    app = Flask(__name__, static_folder=None)
    camera = Camera(camera_index)
    state = {"session": None}
    jobs = {}

    def valid_subject(subject_id):
        if not SUBJECT_ID_PATTERN.match(subject_id or ""):
            abort(400, "Subject ID may only contain letters, digits, '-' and '_' (max 40).")
        return subject_id

    def current_session():
        session = state["session"]
        if session is None or session.finished:
            abort(409, "No session is running.")
        return session

    @app.errorhandler(400)
    @app.errorhandler(404)
    @app.errorhandler(409)
    @app.errorhandler(500)
    def json_error(error):
        return jsonify(error=getattr(error, "description", str(error))), error.code

    @app.get("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.get("/static/<path:name>")
    def static_file(name):
        return send_from_directory(STATIC_DIR, name)

    @app.get("/api/clock")
    def clock_now():
        return jsonify(t=clock())

    @app.get("/api/camera/status")
    def camera_status():
        camera.ensure_running()
        return jsonify(camera.status())

    @app.post("/api/camera/select")
    def camera_select():
        try:
            camera.switch(int(request.json["index"]))
        except RuntimeError as error:
            abort(409, str(error))
        return jsonify(camera.status())

    @app.get("/api/camera/stream")
    def camera_stream():
        camera.ensure_running()
        return Response(
            camera.mjpeg_stream(), mimetype="multipart/x-mixed-replace; boundary=frame"
        )

    @app.post("/api/session/check")
    def session_check():
        subject_id = valid_subject(request.json.get("subject_id"))
        existing = [s for s, present in session_files(logs_dir, subject_id).items() if present]
        return jsonify(existing=existing)

    @app.post("/api/session/start")
    def session_start():
        body = request.json
        subject_id = valid_subject(body.get("subject_id"))
        previous = state["session"]
        if previous is not None and not previous.finished:
            previous.finish(True, {"note": "superseded by a new session"})
        existing = [s for s, present in session_files(logs_dir, subject_id).items() if present]
        if existing and not body.get("overwrite"):
            abort(409, f"Files already exist for {subject_id}: {', '.join(existing)}")
        for suffix in existing:
            (logs_dir / f"{subject_id}{suffix}").unlink()
        meta = {
            "app_version": APP_VERSION,
            "subject_id": subject_id,
            "session": body.get("session"),
            "notes": body.get("notes", ""),
            "started_at": datetime.now().isoformat(),
            "settings": body.get("settings", {}),
            "tasks": body.get("tasks", []),
            "clock_sync_start": body.get("clock_sync"),
            "user_agent": request.headers.get("User-Agent"),
            "camera_index": camera.index,
            "record_webcam": bool(body.get("record_webcam", True)),
        }
        try:
            state["session"] = Session(
                logs_dir, subject_id, meta, camera, meta["record_webcam"]
            )
        except RuntimeError as error:
            abort(500, f"Webcam failed to start: {error}")
        return jsonify(ok=True)

    @app.post("/api/session/marker")
    def session_marker():
        body = request.json
        current_session().marker(str(body["label"]), float(body["t"]))
        return jsonify(ok=True)

    @app.post("/api/session/results")
    def session_results():
        body = request.json
        if body.get("task") not in TASK_COLUMNS:
            abort(400, "Unknown task")
        current_session().save_task(body["task"], body.get("rows", []))
        return jsonify(ok=True)

    @app.post("/api/session/finish")
    def session_finish():
        body = request.get_json(force=True, silent=True) or {}
        session = state["session"]
        if session is None or session.finished:
            return jsonify(ok=True, meta=None)
        meta = session.finish(body.get("aborted", False), {"clock_sync_end": body.get("clock_sync")})
        return jsonify(ok=True, meta=meta)

    @app.get("/api/sessions")
    def sessions_list():
        rows = []
        for subject_id in known_subject_ids(logs_dir):
            meta = read_meta(logs_dir, subject_id)
            files = session_files(logs_dir, subject_id)
            rows.append(
                {
                    "subject_id": subject_id,
                    "session": meta.get("session"),
                    "started_at": meta.get("started_at"),
                    "aborted": meta.get("aborted"),
                    "has_video": files["_webcam.mp4"],
                    "has_trace": files["_pupil_trace.csv"],
                    "source": "app" if meta else "psychopy",
                }
            )
        rows.sort(key=lambda row: row["started_at"] or "", reverse=True)
        return jsonify(sessions=rows, logs_dir=str(logs_dir))

    @app.get("/api/sessions/<subject_id>")
    def session_detail(subject_id):
        valid_subject(subject_id)
        files = session_files(logs_dir, subject_id)
        if not any(files.values()):
            abort(404, "No such session")
        trial_table = logs_dir / f"{subject_id}_trial_pupil_table.csv"
        trial_counts = None
        if trial_table.exists():
            table = pd.read_csv(trial_table)
            trial_counts = table.groupby("task_name")["trial_id"].nunique().to_dict()
        return jsonify(
            subject_id=subject_id,
            meta=read_meta(logs_dir, subject_id),
            files={suffix: present for suffix, present in files.items()},
            summary=scoring.session_summary(logs_dir, subject_id),
            trial_counts=trial_counts,
            job=jobs.get(subject_id),
        )

    @app.get("/api/sessions/<subject_id>/trace")
    def session_trace(subject_id):
        valid_subject(subject_id)
        trace_path = logs_dir / f"{subject_id}_pupil_trace.csv"
        if not trace_path.exists():
            abort(404, "No pupil trace yet; run the analysis first.")
        trace = pd.read_csv(trace_path).sort_values("psychopy_clock_time")
        times = trace["psychopy_clock_time"].to_numpy(dtype=float)
        origin = float(times[0]) if len(times) else 0.0
        centers, values = binned_trace(
            times - origin, trace["pupil_diameter_px"].to_numpy(dtype=float)
        )
        intervals = []
        markers_path = logs_dir / f"{subject_id}_markers.csv"
        if markers_path.exists():
            intervals = [
                {"task": name, "start": round(start - origin, 3), "end": round(end - origin, 3)}
                for name, start, end in scoring.task_intervals(pd.read_csv(markers_path))
            ]
        detected = trace["pupil_diameter_px"].notna()
        return jsonify(
            t=centers,
            diameter=values,
            intervals=intervals,
            frames=int(len(trace)),
            detect_rate=float(detected.mean()) if len(trace) else None,
        )

    def run_analysis(subject_id):
        from pupilad.webcam.trial_table import build_trial_table
        from pupilad.webcam.video_extraction import extract_pupil_trace

        job = jobs[subject_id]
        try:
            timestamps = pd.read_csv(logs_dir / f"{subject_id}_webcam_frame_timestamps.csv")
            total = max(len(timestamps), 1)

            def progress(done):
                job["progress"] = done / total

            job["message"] = "Detecting pupils in video frames"
            trace = extract_pupil_trace(
                logs_dir / f"{subject_id}_webcam.mp4", timestamps, subject_id, progress
            )
            trace.to_csv(logs_dir / f"{subject_id}_pupil_trace.csv", index=False)
            job["progress"] = 1.0
            job["message"] = "Matching trials to pupil samples"
            try:
                table = build_trial_table(logs_dir, subject_id)
                table.to_csv(logs_dir / f"{subject_id}_trial_pupil_table.csv", index=False)
                detail = f"{table['trial_id'].nunique()} trials matched"
            except SystemExit as reason:
                detail = f"Trial table skipped: {reason}"
            detected = trace["pupil_diameter_px"].notna().mean() if len(trace) else 0.0
            job.update(
                state="done",
                message=f"Pupil found in {detected:.0%} of {len(trace)} frames. {detail}.",
            )
        except Exception as error:
            job.update(state="error", message=str(error))

    @app.post("/api/sessions/<subject_id>/analyze")
    def session_analyze(subject_id):
        valid_subject(subject_id)
        files = session_files(logs_dir, subject_id)
        if not (files["_webcam.mp4"] and files["_webcam_frame_timestamps.csv"]):
            abort(409, "This session has no webcam recording to analyze.")
        session = state["session"]
        if session is not None and not session.finished and session.subject_id == subject_id:
            abort(409, "This session is still recording.")
        if jobs.get(subject_id, {}).get("state") == "running":
            return jsonify(jobs[subject_id])
        jobs[subject_id] = {"state": "running", "progress": 0.0, "message": "Starting"}
        threading.Thread(target=run_analysis, args=(subject_id,), daemon=True).start()
        return jsonify(jobs[subject_id])

    @app.get("/api/sessions/<subject_id>/job")
    def session_job(subject_id):
        return jsonify(jobs.get(subject_id))

    @app.get("/files/<path:name>")
    def download(name):
        if "/" in name or "\\" in name:
            abort(404)
        return send_from_directory(logs_dir, name, as_attachment=True)

    app.config["camera"] = camera
    return app


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pupilad app")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--logs_dir", type=Path, default=DEFAULT_LOGS_DIR)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    app = create_app(args.logs_dir.resolve(), args.camera)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"PupilAD app running at {url}  (data folder: {args.logs_dir.resolve()})")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    try:
        app.run(host="127.0.0.1", port=args.port, threaded=True, debug=False)
    finally:
        app.config["camera"].shutdown()


if __name__ == "__main__":
    main()
