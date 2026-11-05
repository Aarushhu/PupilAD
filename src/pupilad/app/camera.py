import csv
import sys
import threading
import time
from collections import deque
from datetime import datetime

import cv2

from pupilad.webcam.pupil_detector import detect_pupil_diameter_px

clock = time.perf_counter

PREVIEW_EVERY_N_FRAMES = 2
DETECT_INTERVAL_S = 0.5
JPEG_QUALITY = 70
START_TIMEOUT_S = 20.0
CAPTURE_BACKEND = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY


class Camera:
    def __init__(self, index=0, target_fps=30):
        self.index = index
        self.target_fps = target_fps
        self._lock = threading.Lock()
        self._new_frame = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._thread = None
        self._recording = None
        self.error = None
        self.jpeg = None
        self.jpeg_id = 0
        self.frame_size = None
        self._frame_times = deque(maxlen=30)
        self._detections = deque(maxlen=20)
        self.last_diameter_px = None

    def ensure_running(self):
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self.error = None
            self.frame_size = None
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def wait_until_ready(self, timeout_s=START_TIMEOUT_S):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.error is not None:
                raise RuntimeError(self.error)
            if self.frame_size is not None:
                return
            time.sleep(0.05)
        raise RuntimeError(f"Camera {self.index} did not deliver a frame within {timeout_s:.0f} s")

    def _run(self):
        try:
            self._capture_loop()
        except Exception as error:
            self.error = f"Camera thread stopped: {type(error).__name__}: {error}"

    def _detect(self, frame):
        try:
            return detect_pupil_diameter_px(frame)
        except cv2.error:
            return None

    def _capture_loop(self):
        capture = cv2.VideoCapture(self.index, CAPTURE_BACKEND)
        if not capture.isOpened():
            self.error = f"Could not open camera index {self.index}"
            return
        frame_count = 0
        last_detect = 0.0
        try:
            while not self._stop.is_set():
                ok, frame = capture.read()
                if not ok:
                    time.sleep(0.005)
                    continue
                now = clock()
                wall = datetime.now().isoformat()
                frame_count += 1
                self._frame_times.append(now)
                with self._lock:
                    if self.frame_size is None:
                        self.frame_size = (frame.shape[1], frame.shape[0])
                    recording = self._recording
                    if recording is not None:
                        self._write_frame(recording, frame, wall, now)
                if recording is not None:
                    continue
                if now - last_detect > DETECT_INTERVAL_S:
                    last_detect = now
                    self.last_diameter_px = self._detect(frame)
                    self._detections.append(self.last_diameter_px is not None)
                if frame_count % PREVIEW_EVERY_N_FRAMES == 0:
                    ok, encoded = cv2.imencode(
                        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                    )
                    if ok:
                        with self._new_frame:
                            self.jpeg = encoded.tobytes()
                            self.jpeg_id += 1
                            self._new_frame.notify_all()
        finally:
            with self._lock:
                if self._recording is not None and self._recording["writer"] is not None:
                    self._recording["writer"].release()
                    self._recording["writer"] = None
            capture.release()

    def _write_frame(self, recording, frame, wall, now):
        if recording["writer"] is None:
            height, width = frame.shape[:2]
            recording["writer"] = cv2.VideoWriter(
                str(recording["video_path"]),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.target_fps,
                (width, height),
            )
        recording["writer"].write(frame)
        recording["log"].append((len(recording["log"]), wall, now))

    def start_recording(self, video_path, timestamp_path):
        self.ensure_running()
        self.wait_until_ready()
        with self._lock:
            if self._recording is not None:
                raise RuntimeError("Camera is already recording")
            self._recording = {
                "video_path": video_path,
                "timestamp_path": timestamp_path,
                "writer": None,
                "log": [],
            }

    def stop_recording(self):
        with self._lock:
            recording, self._recording = self._recording, None
        if recording is None:
            return None
        if recording["writer"] is not None:
            recording["writer"].release()
        log = recording["log"]
        with open(recording["timestamp_path"], "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["frame_index", "wall_clock_time", "psychopy_clock_time"])
            writer.writerows(log)
        duration = log[-1][2] - log[0][2] if len(log) > 1 else 0.0
        return {
            "frames": len(log),
            "duration_s": duration,
            "mean_fps": (len(log) - 1) / duration if duration > 0 else None,
        }

    @property
    def recording(self):
        return self._recording is not None

    def status(self):
        times = list(self._frame_times)
        running = self._thread is not None and self._thread.is_alive()
        fps = None
        if running and len(times) > 1 and clock() - times[-1] < 1.0:
            fps = (len(times) - 1) / (times[-1] - times[0])
        detections = list(self._detections)
        return {
            "index": self.index,
            "running": running,
            "error": self.error,
            "recording": self.recording,
            "frame_size": self.frame_size,
            "fps": fps,
            "pupil_detect_rate": sum(detections) / len(detections) if detections else None,
            "last_diameter_px": self.last_diameter_px,
        }

    def mjpeg_stream(self):
        last_sent = -1
        while not self._stop.is_set():
            with self._new_frame:
                self._new_frame.wait(timeout=1.0)
                jpeg, jpeg_id = self.jpeg, self.jpeg_id
            if self.error is not None and jpeg is None:
                return
            if jpeg is None or jpeg_id == last_sent:
                continue
            last_sent = jpeg_id
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"

    def switch(self, index):
        if self.recording:
            raise RuntimeError("Cannot switch cameras while recording")
        self.shutdown()
        self.index = index
        self.jpeg = None
        self._frame_times.clear()
        self._detections.clear()
        self.last_diameter_px = None
        self.ensure_running()

    def shutdown(self):
        self._stop.set()
        with self._new_frame:
            self._new_frame.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._thread = None
