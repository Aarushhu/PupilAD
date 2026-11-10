import cv2
import numpy as np

FACE_SCALE_FACTOR = 1.3
FACE_MIN_NEIGHBORS = 5
FACE_MIN_SIZE = (100, 100)
EYE_SCALE_FACTOR = 1.1
EYE_MIN_NEIGHBORS = 5
EYE_MAX_CENTER_FRACTION = 0.55
EYES_PER_FACE = 2
CORE_ROWS = (0.25, 0.85)
CORE_COLS = (0.15, 0.85)
MIN_EYE_CONTRAST = 12
DARK_FRACTION = 0.3
MIN_DARK_STEP = 6
MIN_PUPIL_AREA_PX = 6
MAX_PUPIL_AREA_FRACTION = 0.3
MAX_PUPIL_ASPECT = 2.2

_face_cascade = None
_eye_cascade = None


def cascades():
    global _face_cascade, _eye_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        _eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    return _face_cascade, _eye_cascade


def equivalent_diameter(area):
    return 2 * np.sqrt(area / np.pi)


def detect_faces(gray_frame, min_size=FACE_MIN_SIZE):
    face_cascade, _ = cascades()
    faces = face_cascade.detectMultiScale(
        gray_frame, FACE_SCALE_FACTOR, FACE_MIN_NEIGHBORS, minSize=min_size
    )
    return sorted((tuple(int(v) for v in box) for box in faces), key=lambda b: -b[2] * b[3])


def detect_eyes(face_gray):
    _, eye_cascade = cascades()
    face_h = face_gray.shape[0]
    eyes = [
        tuple(int(v) for v in box)
        for box in eye_cascade.detectMultiScale(face_gray, EYE_SCALE_FACTOR, EYE_MIN_NEIGHBORS)
        if box[1] + box[3] / 2 < face_h * EYE_MAX_CENTER_FRACTION
    ]
    return sorted(eyes, key=lambda b: -b[2] * b[3])[:EYES_PER_FACE]


def eye_core(eye_gray):
    height, width = eye_gray.shape
    return eye_gray[
        int(height * CORE_ROWS[0]) : int(height * CORE_ROWS[1]),
        int(width * CORE_COLS[0]) : int(width * CORE_COLS[1]),
    ]


def eye_contrast(eye_gray):
    blurred = cv2.GaussianBlur(eye_core(eye_gray), (5, 5), 0)
    return float(np.median(blurred)) - float(blurred.min())


def pupil_area(eye_gray):
    core = eye_core(eye_gray)
    if core.size == 0:
        return 0.0
    blurred = cv2.GaussianBlur(core, (5, 5), 0)
    darkest = float(blurred.min())
    median = float(np.median(blurred))
    if median - darkest < MIN_EYE_CONTRAST:
        return 0.0
    threshold = darkest + max(MIN_DARK_STEP, DARK_FRACTION * (median - darkest))
    mask = (blurred <= threshold).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    row, col = np.unravel_index(np.argmin(blurred), blurred.shape)
    label = labels[row, col]
    area = stats[label, cv2.CC_STAT_AREA]
    width = stats[label, cv2.CC_STAT_WIDTH]
    height = stats[label, cv2.CC_STAT_HEIGHT]
    if area < MIN_PUPIL_AREA_PX or area > MAX_PUPIL_AREA_FRACTION * core.size:
        return 0.0
    if max(width, height) > MAX_PUPIL_ASPECT * min(width, height):
        return 0.0
    return float(area)


def eye_regions(gray_frame):
    faces = detect_faces(gray_frame)
    if not faces:
        return
    face_x, face_y, face_w, face_h = faces[0]
    face = gray_frame[face_y : face_y + face_h, face_x : face_x + face_w]
    for eye_x, eye_y, eye_w, eye_h in detect_eyes(face):
        yield face[eye_y : eye_y + eye_h, eye_x : eye_x + eye_w]


def detect_pupil_diameter_px(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diameters = [
        equivalent_diameter(area)
        for area in (pupil_area(region) for region in eye_regions(gray))
        if area > 0
    ]
    return float(np.mean(diameters)) if diameters else None
