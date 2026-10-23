"""
PupilAD — Offline Pupil Extraction from Webcam Video
=====================================================

run_task_battery.py's webcam mode saves a raw {subject_id}_webcam.mp4 plus a
per-frame timestamp log ({subject_id}_webcam_frame_timestamps.csv) and does
NOT extract pupil diameter live — extraction happens afterward, on the saved
video. This script is that missing step.

It reuses detect_pupil() from src/pupil_tracker.py directly (loaded by file
path, not reimplemented) so this stays in sync with that file rather than
drifting into a second copy of the same detection logic.

IMPORTANT — these are PIXEL diameters (2*sqrt(contour_area/pi) from a
thresholded Haar-cascade eye ROI), NOT calibrated millimeters. There is no
calibration step anywhere in this pipeline (no known reference object or
fixed camera distance), so absolute magnitudes are not physically meaningful
and should not be checked against literature mm cutoffs — see
validate_lc_score.py's Layer 1 physiological-plausibility check, which
assumes real mm and does not apply as-is to this data source. Within-subject,
rank-based comparisons (does dilation increase with load, does it differ
correct vs. incorrect) remain meaningful, since they don't depend on
absolute calibration.

Usage
-----
    .venv-psychopy/bin/python task_battery/extract_pupil_from_video.py \\
        --subject_id S001 \\
        --logs_dir task_battery/task_battery_logs

Requires opencv-python, which lives in .venv-psychopy/ (not .venv/) in this
project — run this with that interpreter.

NOTE: pip's latest opencv-python (5.0.0.93 as of writing) silently ships
without cv2.CascadeClassifier at all — this fails with
"AttributeError: module 'cv2' has no attribute 'CascadeClassifier'" the
first time detect_pupil() is called. Pin to a 4.x release instead:
    .venv-psychopy/bin/pip install "opencv-python==4.10.0.84"
"""

import argparse
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_pupil_tracker():
    spec = importlib.util.spec_from_file_location("pupilad_pupil_tracker", REPO_ROOT / "src" / "pupil_tracker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_pupil_trace(video_path: Path, frame_timestamps_csv: Path, subject_id: str) -> pd.DataFrame:
    """
    Walk the recorded video frame-by-frame in write order, which matches the
    frame_index order in the timestamp log 1:1 (WebcamRecorder appends to its
    frame log in the same loop iteration it writes each frame — see
    run_task_battery.py's WebcamRecorder._record_loop), and run the same
    Haar-cascade pupil detector used live in src/pupil_tracker.py.
    """
    tracker = _load_pupil_tracker()
    timestamps = pd.read_csv(frame_timestamps_csv)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")

    rows = []
    frame_idx = 0
    n_timestamp_rows = len(timestamps)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx >= n_timestamp_rows:
                print(
                    f"[warn] video has more frames than logged timestamps "
                    f"({frame_idx} vs {n_timestamp_rows}) — stopping at the timestamp log's length."
                )
                break

            diameter_px = tracker.detect_pupil(frame)
            ts_row = timestamps.iloc[frame_idx]
            rows.append({
                "subject_id": subject_id,
                "frame_index": int(ts_row["frame_index"]),
                "wall_clock_time": ts_row["wall_clock_time"],
                "psychopy_clock_time": float(ts_row["psychopy_clock_time"]),
                "pupil_diameter_px": diameter_px if diameter_px is not None else np.nan,
            })
            frame_idx += 1
    finally:
        cap.release()

    if frame_idx < n_timestamp_rows:
        print(
            f"[warn] timestamp log has more rows than the video had frames "
            f"({n_timestamp_rows} vs {frame_idx}) — some logged frames were never written to disk."
        )

    df = pd.DataFrame(rows)
    frac_detected = df["pupil_diameter_px"].notna().mean() if len(df) else 0.0
    print(f"Processed {len(df)} frames — pupil detected in {frac_detected * 100:.1f}% of them.")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Extract a pupil-diameter-over-time trace from a recorded webcam session."
    )
    parser.add_argument("--subject_id", required=True)
    parser.add_argument("--logs_dir", default="task_battery/task_battery_logs")
    parser.add_argument("--video", default=None,
                         help="Override path to the .mp4 (default: {logs_dir}/{subject_id}_webcam.mp4)")
    parser.add_argument("--frame_timestamps", default=None,
                         help="Override path to the frame-timestamps CSV")
    parser.add_argument("--output", default=None,
                         help="Output CSV path (default: {logs_dir}/{subject_id}_pupil_trace.csv)")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    video_path = Path(args.video) if args.video else logs_dir / f"{args.subject_id}_webcam.mp4"
    ts_path = Path(args.frame_timestamps) if args.frame_timestamps else logs_dir / f"{args.subject_id}_webcam_frame_timestamps.csv"
    output_path = Path(args.output) if args.output else logs_dir / f"{args.subject_id}_pupil_trace.csv"

    df = extract_pupil_trace(video_path, ts_path, args.subject_id)
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
