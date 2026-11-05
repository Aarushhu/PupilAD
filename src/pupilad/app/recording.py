import csv
from datetime import datetime, timedelta

import cv2

NOMINAL_FPS = 30.0


def convert_recording(raw_path, video_path, timestamps_path, start_s, start_epoch_ms, duration_s=None, progress=None):
    capture = cv2.VideoCapture(str(raw_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open the uploaded recording {raw_path.name}")
    start_wall = datetime.fromtimestamp(start_epoch_ms / 1000)
    writer = None
    rows = []
    out_of_order = 0
    last_ms = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            ms = capture.get(cv2.CAP_PROP_POS_MSEC)
            if last_ms is not None and ms <= last_ms:
                out_of_order += 1
            last_ms = ms
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), NOMINAL_FPS, (width, height)
                )
            writer.write(frame)
            wall = start_wall + timedelta(milliseconds=ms)
            rows.append((len(rows), wall.isoformat(), round(start_s + ms / 1000, 6)))
            if progress is not None and duration_s:
                progress(min(1.0, ms / 1000 / duration_s))
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if not rows:
        raise RuntimeError("The uploaded recording contains no video frames")
    with open(timestamps_path, "w", newline="") as handle:
        out = csv.writer(handle)
        out.writerow(["frame_index", "wall_clock_time", "psychopy_clock_time"])
        out.writerows(rows)
    duration = rows[-1][2] - rows[0][2]
    return {
        "frames": len(rows),
        "duration_s": round(duration, 3),
        "mean_fps": round((len(rows) - 1) / duration, 2) if duration > 0 else None,
        "frame_size": [int(width), int(height)],
        "out_of_order_timestamps": out_of_order,
    }
