import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from pupilad.paths import DATA_DIR
from pupilad.webcam.pupil_detector import detect_pupil_diameter_px

DEFAULT_LOGS_DIR = DATA_DIR / "task_battery_logs"


def load_webcam_frame_timestamps(path):
    return pd.read_csv(path)


def video_frames(video_path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video {video_path}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                return
            yield frame
    finally:
        capture.release()


def extract_pupil_trace(video_path, frame_timestamps, subject_id):
    rows = []
    for frame, (_, timestamp) in zip(video_frames(video_path), frame_timestamps.iterrows()):
        diameter_px = detect_pupil_diameter_px(frame)
        rows.append(
            {
                "subject_id": subject_id,
                "frame_index": int(timestamp["frame_index"]),
                "wall_clock_time": timestamp["wall_clock_time"],
                "psychopy_clock_time": float(timestamp["psychopy_clock_time"]),
                "pupil_diameter_px": np.nan if diameter_px is None else diameter_px,
            }
        )
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pupilad extract-webcam")
    parser.add_argument("--subject_id", required=True)
    parser.add_argument("--logs_dir", type=Path, default=DEFAULT_LOGS_DIR)
    args = parser.parse_args(argv)
    video_path = args.logs_dir / f"{args.subject_id}_webcam.mp4"
    timestamps = load_webcam_frame_timestamps(
        args.logs_dir / f"{args.subject_id}_webcam_frame_timestamps.csv"
    )
    trace = extract_pupil_trace(video_path, timestamps, args.subject_id)
    output_path = args.logs_dir / f"{args.subject_id}_pupil_trace.csv"
    trace.to_csv(output_path, index=False)
    detected = trace["pupil_diameter_px"].notna().mean() if len(trace) else 0.0
    print(
        f"{len(trace)} of {len(timestamps)} logged frames processed, "
        f"pupil found in {detected:.1%}; wrote {output_path}"
    )


if __name__ == "__main__":
    main()
