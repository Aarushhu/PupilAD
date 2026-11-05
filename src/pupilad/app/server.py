import argparse
import csv
import json
import re
import secrets
import threading
import webbrowser
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, abort, jsonify, redirect, request, send_from_directory

from pupilad.app import scoring
from pupilad.app.checks import inspect_frame
from pupilad.app.recording import convert_recording
from pupilad.paths import DATA_DIR

DEFAULT_LOGS_DIR = DATA_DIR / "task_battery_logs"
CERT_DIR = DATA_DIR / "app_certificate"
STATIC_DIR = Path(__file__).with_name("static")
SUBJECT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
APP_VERSION = "2"
MAX_TRACE_POINTS = 1500
ACCESS_COOKIE = "pupilad_key"
LOOPBACK = {"127.0.0.1", "::1"}

TASK_COLUMNS = {
    "nback": ["trial", "letter", "is_match", "responded", "rt", "correct", "onset_time", "response"],
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
MEDIA_KINDS = {"video": "_webcam_raw", "audio": "_fluency_audio"}
MEDIA_EXTENSIONS = (".webm", ".mp4")
SESSION_FILE_SUFFIXES = [
    "_session.json",
    "_markers.csv",
    "_nback.csv",
    "_digitspan.csv",
    "_stroop.csv",
    "_webcam_raw.webm",
    "_webcam_raw.mp4",
    "_webcam.mp4",
    "_webcam_frame_timestamps.csv",
    "_fluency_audio.webm",
    "_fluency_audio.mp4",
    "_pupil_trace.csv",
    "_trial_pupil_table.csv",
]


def media_extension(mime):
    return ".mp4" if "mp4" in str(mime or "").lower() else ".webm"


def epoch_to_iso(epoch_ms):
    return datetime.fromtimestamp(epoch_ms / 1000).isoformat()


class Session:
    def __init__(self, logs_dir, subject_id, meta):
        self.logs_dir = logs_dir
        self.subject_id = subject_id
        self.meta = meta
        self.lock = threading.Lock()
        self.finished = False
        self.next_seq = {kind: 0 for kind in MEDIA_KINDS}
        self.media_paths = {}
        logs_dir.mkdir(parents=True, exist_ok=True)
        self.marker_handle = open(self.path("_markers.csv"), "w", newline="")
        self.marker_writer = csv.writer(self.marker_handle)
        self.marker_writer.writerow(["wall_clock_time", "psychopy_clock_time", "event"])
        self.write_meta()

    def path(self, suffix):
        return self.logs_dir / f"{self.subject_id}{suffix}"

    def write_meta(self):
        self.path("_session.json").write_text(json.dumps(self.meta, indent=2))

    def marker(self, label, timestamp, epoch_ms):
        with self.lock:
            if self.finished:
                return
            self.marker_writer.writerow([epoch_to_iso(epoch_ms), timestamp, label])
            self.marker_handle.flush()

    def save_task(self, task, rows):
        frame = pd.DataFrame(rows, columns=TASK_COLUMNS[task])
        frame.to_csv(self.path(f"_{task}.csv"), index=False)
        with self.lock:
            self.meta.setdefault("tasks_saved", [])
            if task not in self.meta["tasks_saved"]:
                self.meta["tasks_saved"].append(task)
            self.write_meta()

    def append_media(self, kind, seq, mime, data):
        with self.lock:
            if self.finished:
                abort(409, "The session has already finished.")
            expected = self.next_seq[kind]
            if seq < expected:
                return expected
            if seq > expected:
                abort(409, f"Expected {kind} chunk {expected}, got {seq}.")
            path = self.media_paths.get(kind)
            if path is None:
                path = self.path(MEDIA_KINDS[kind] + media_extension(mime))
                self.media_paths[kind] = path
                path.write_bytes(b"")
            with open(path, "ab") as handle:
                handle.write(data)
            self.next_seq[kind] = expected + 1
            return self.next_seq[kind]

    def finish(self, aborted, extra):
        with self.lock:
            if self.finished:
                return self.meta
            self.finished = True
            self.marker_handle.close()
        self.meta["ended_at"] = datetime.now().isoformat()
        self.meta["aborted"] = bool(aborted)
        self.meta["media_chunks"] = dict(self.next_seq)
        self.meta.update({key: value for key, value in (extra or {}).items() if value is not None})
        self.write_meta()
        return self.meta


def session_files(logs_dir, subject_id):
    return {
        suffix: (logs_dir / f"{subject_id}{suffix}").exists() for suffix in SESSION_FILE_SUFFIXES
    }


def raw_video_path(logs_dir, subject_id):
    for extension in MEDIA_EXTENSIONS:
        path = logs_dir / f"{subject_id}_webcam_raw{extension}"
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


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


def update_meta(logs_dir, subject_id, **values):
    path = logs_dir / f"{subject_id}_session.json"
    meta = read_meta(logs_dir, subject_id)
    if not meta:
        return
    meta.update(values)
    path.write_text(json.dumps(meta, indent=2))


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


def create_app(logs_dir=DEFAULT_LOGS_DIR, access_key=None, lan_url=None, auto_process=True):
    app = Flask(__name__, static_folder=None)
    app.config.update(ACCESS_KEY=access_key, LAN_URL=lan_url)
    state = {"session": None}
    jobs = {}

    def valid_subject(subject_id):
        if not SUBJECT_ID_PATTERN.match(subject_id or ""):
            abort(400, "Participant ID may only contain letters, digits, '-' and '_' (max 40).")
        return subject_id

    def current_session():
        session = state["session"]
        if session is None or session.finished:
            abort(409, "No session is running.")
        return session

    def is_local():
        return request.remote_addr in LOOPBACK

    @app.before_request
    def check_access():
        key = app.config["ACCESS_KEY"]
        if not key or is_local():
            return None
        if secrets.compare_digest(request.cookies.get(ACCESS_COOKIE, ""), key):
            return None
        if secrets.compare_digest(request.args.get("key", ""), key):
            response = redirect(request.path)
            response.set_cookie(
                ACCESS_COOKIE, key, max_age=30 * 86400, httponly=True, secure=True, samesite="Strict"
            )
            return response
        abort(403, "Open the full link shown on the PupilAD computer; it includes an access key.")

    @app.errorhandler(400)
    @app.errorhandler(403)
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
        response = send_from_directory(STATIC_DIR, name)
        response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/api/info")
    def info():
        return jsonify(
            app_version=APP_VERSION,
            lan_url=app.config["LAN_URL"] if is_local() else None,
            logs_dir=str(logs_dir),
        )

    @app.post("/api/camera/check")
    def camera_check():
        data = request.get_data()
        if not data:
            abort(400, "No image received.")
        try:
            return jsonify(inspect_frame(data))
        except ValueError as error:
            abort(400, str(error))

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
        if jobs.get(subject_id, {}).get("state") == "running":
            abort(409, f"{subject_id} is still being processed; wait for it to finish.")
        for suffix in existing:
            (logs_dir / f"{subject_id}{suffix}").unlink()
        jobs.pop(subject_id, None)
        meta = {
            "app_version": APP_VERSION,
            "subject_id": subject_id,
            "session": body.get("session"),
            "notes": body.get("notes", ""),
            "started_at": datetime.now().isoformat(),
            "settings": body.get("settings", {}),
            "tasks": body.get("tasks", []),
            "device": body.get("device"),
            "user_agent": request.headers.get("User-Agent"),
            "record_webcam": bool(body.get("record_webcam", True)),
            "clock": "browser performance.now() in seconds",
        }
        state["session"] = Session(logs_dir, subject_id, meta)
        return jsonify(ok=True)

    @app.post("/api/session/marker")
    def session_marker():
        body = request.json
        current_session().marker(str(body["label"]), float(body["t"]), float(body["epoch_ms"]))
        return jsonify(ok=True)

    @app.post("/api/session/results")
    def session_results():
        body = request.json
        if body.get("task") not in TASK_COLUMNS:
            abort(400, "Unknown task")
        current_session().save_task(body["task"], body.get("rows", []))
        return jsonify(ok=True)

    @app.post("/api/session/media/<kind>")
    def session_media(kind):
        if kind not in MEDIA_KINDS:
            abort(404)
        try:
            seq = int(request.args["seq"])
        except (KeyError, ValueError):
            abort(400, "Missing chunk number.")
        next_seq = current_session().append_media(
            kind, seq, request.args.get("mime"), request.get_data()
        )
        return jsonify(ok=True, next=next_seq)

    @app.post("/api/session/finish")
    def session_finish():
        body = request.get_json(force=True, silent=True) or {}
        session = state["session"]
        if session is None or session.finished:
            return jsonify(ok=True, meta=None)
        meta = session.finish(
            body.get("aborted", False),
            {
                "recording": body.get("recording"),
                "audio": body.get("audio"),
                "practice": body.get("practice"),
                "events": body.get("events"),
            },
        )
        if auto_process and raw_video_path(logs_dir, session.subject_id) is not None:
            start_job(session.subject_id)
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
                    "device": (meta.get("device") or {}).get("kind"),
                    "has_video": files["_webcam.mp4"] or raw_video_path(logs_dir, subject_id) is not None,
                    "has_trace": files["_pupil_trace.csv"],
                    "pupil_detect_rate": meta.get("pupil_detect_rate"),
                    "processing": jobs.get(subject_id, {}).get("state") == "running",
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

    def prepare_video(subject_id, job):
        raw = raw_video_path(logs_dir, subject_id)
        if raw is None:
            return
        recording = read_meta(logs_dir, subject_id).get("recording") or {}
        if "start_s" not in recording or "start_epoch_ms" not in recording:
            raise RuntimeError("The session has no recording start time, so the video can't be aligned.")
        job.update(step="video", message="Preparing the video", progress=0.0)

        def progress(fraction):
            job["progress"] = fraction

        stats = convert_recording(
            raw,
            logs_dir / f"{subject_id}_webcam.mp4",
            logs_dir / f"{subject_id}_webcam_frame_timestamps.csv",
            float(recording["start_s"]),
            float(recording["start_epoch_ms"]),
            recording.get("duration_s"),
            progress,
        )
        update_meta(logs_dir, subject_id, webcam=stats)

    def extract_pupils(subject_id, job):
        from pupilad.webcam.trial_table import build_trial_table
        from pupilad.webcam.video_extraction import extract_pupil_trace

        timestamps = pd.read_csv(logs_dir / f"{subject_id}_webcam_frame_timestamps.csv")
        total = max(len(timestamps), 1)

        def progress(done):
            job["progress"] = done / total

        job.update(step="pupil", message="Finding the pupil in each video frame", progress=0.0)
        trace = extract_pupil_trace(
            logs_dir / f"{subject_id}_webcam.mp4", timestamps, subject_id, progress
        )
        trace.to_csv(logs_dir / f"{subject_id}_pupil_trace.csv", index=False)
        job.update(progress=1.0, message="Matching trials to pupil samples")
        try:
            table = build_trial_table(logs_dir, subject_id)
            table.to_csv(logs_dir / f"{subject_id}_trial_pupil_table.csv", index=False)
            detail = f"{table['trial_id'].nunique()} trials matched"
        except SystemExit as reason:
            detail = f"Trial table skipped: {reason}"
        detected = float(trace["pupil_diameter_px"].notna().mean()) if len(trace) else 0.0
        update_meta(logs_dir, subject_id, pupil_detect_rate=round(detected, 4))
        return f"Pupil found in {detected:.0%} of {len(trace)} frames. {detail}."

    def run_job(subject_id, force_video):
        job = jobs[subject_id]
        try:
            mp4 = logs_dir / f"{subject_id}_webcam.mp4"
            if force_video or not mp4.exists():
                prepare_video(subject_id, job)
            if not mp4.exists():
                raise RuntimeError("This session has no webcam recording to analyze.")
            message = extract_pupils(subject_id, job)
            job.update(state="done", step=None, message=message)
        except Exception as error:
            job.update(state="error", step=None, message=str(error))

    def start_job(subject_id, force_video=False):
        if jobs.get(subject_id, {}).get("state") == "running":
            return jobs[subject_id]
        jobs[subject_id] = {"state": "running", "step": "start", "progress": 0.0, "message": "Starting"}
        threading.Thread(target=run_job, args=(subject_id, force_video), daemon=True).start()
        return jobs[subject_id]

    @app.post("/api/sessions/<subject_id>/analyze")
    def session_analyze(subject_id):
        valid_subject(subject_id)
        files = session_files(logs_dir, subject_id)
        has_raw = raw_video_path(logs_dir, subject_id) is not None
        if not (has_raw or (files["_webcam.mp4"] and files["_webcam_frame_timestamps.csv"])):
            abort(409, "This session has no webcam recording to analyze.")
        session = state["session"]
        if session is not None and not session.finished and session.subject_id == subject_id:
            abort(409, "This session is still recording.")
        return jsonify(start_job(subject_id))

    @app.get("/api/sessions/<subject_id>/job")
    def session_job(subject_id):
        return jsonify(jobs.get(subject_id))

    @app.get("/files/<path:name>")
    def download(name):
        if "/" in name or "\\" in name:
            abort(404)
        return send_from_directory(logs_dir, name, as_attachment=True)

    return app


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pupilad app")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--logs_dir", type=Path, default=DEFAULT_LOGS_DIR)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--lan",
        action="store_true",
        help="also serve phones and tablets on this Wi-Fi network over HTTPS, protected by an access key",
    )
    args = parser.parse_args(argv)
    logs_dir = args.logs_dir.resolve()
    ssl_context = None
    access_key = None
    lan_url = None
    host = "127.0.0.1"
    local_url = f"http://127.0.0.1:{args.port}/"
    if args.lan:
        from pupilad.app.tls import ensure_certificate, lan_ip

        ip = lan_ip()
        ssl_context = tuple(str(path) for path in ensure_certificate(CERT_DIR, ip))
        access_key = secrets.token_urlsafe(9)
        lan_url = f"https://{ip}:{args.port}/?key={access_key}"
        host = "0.0.0.0"
        local_url = f"https://127.0.0.1:{args.port}/"
    app = create_app(logs_dir, access_key=access_key, lan_url=lan_url)
    print(f"PupilAD app running at {local_url}  (data folder: {logs_dir})")
    if lan_url:
        print(f"On a phone or tablet on the same Wi-Fi, open: {lan_url}")
        print("The browser will warn that the certificate is self-signed; choose to continue.")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, args=(local_url,)).start()
    app.run(host=host, port=args.port, threaded=True, debug=False, ssl_context=ssl_context)


if __name__ == "__main__":
    main()
